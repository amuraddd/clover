"""DDPO: denoising diffusion policy optimization.

Converted from ``clover/exp/ddpo.ipynb``. Run with
``python -m clover.baselines.ddpo``.
"""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from tqdm.auto import trange

from clover.baselines.common import (
    evaluate,
    make_reward_fn,
    normalize_rewards,
    parse_config,
    prepare_output,
    save_evaluation_metrics,
    save_trajectory_data,
    save_training_data,
)
from clover.utils.prompts import DEFAULT_EVAL_PROMPTS, DEFAULT_TRAIN_PROMPTS, B2_FULL_TRAIN_PROMPTS, B2_FULL_EVAL_PROMPTS
from clover.utils.baseline_utils import (
    decode_latents,
    ddpm_step_with_log_prob,
    encode_prompts,
    load_lora_pipeline,
    ppo_update,
    predict_noise,
    resolve_gpu_ids,
    sample_prompt_batch,
    save_json,
    save_lora_weights,
    set_seed,
    trainable_parameters,
    unet_config,
)


@dataclass
class DDPOConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    output_dir: str = "outputs/ddpo"
    seed: int = 17
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    use_data_parallel: bool = False
    prompt: str = "a colorful clover field at sunrise, high detail"
    negative_prompt: str = "blurry, low quality, distorted"
    train_prompts: tuple[str, ...] = B2_FULL_TRAIN_PROMPTS
    eval_prompts: tuple[str, ...] = B2_FULL_EVAL_PROMPTS
    reward_type: str = "clip"  # Options: "aesthetic" (more to come)
    height: int = 512
    width: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    eta: float = 1.0
    rollouts_per_epoch: int = 256
    train_epochs: int = 10
    ppo_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 1e-9
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 2
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("to_v", "to_k", "to_q")
    clip_range: float = 1e-4
    ppo_log_ratio_clip: float = 2.0
    max_grad_norm: float = 0.1
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    log_every: int = 1
    save_every: int = 5
    evaluate_every: int = 2


@torch.no_grad()
def collect_rollouts(
    pipe: Any,
    batch_size: int,
    config: DDPOConfig,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    reward_fn: Any,
    vae_scale_factor: int,
) -> dict[str, Tensor | list[str] | list[Image.Image]]:
    pipe.unet.eval()
    prompts = sample_prompt_batch(config.train_prompts, batch_size)
    prompt_embeds = encode_prompts(pipe, prompts, config.negative_prompt, device, dtype)
    pipe.scheduler.set_timesteps(config.num_inference_steps, device=device)
    latent_shape = (
        batch_size,
        unet_config(pipe).in_channels,
        config.height // vae_scale_factor,
        config.width // vae_scale_factor,
    )
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
    latents = latents * pipe.scheduler.init_noise_sigma
    states, actions, log_probs, timesteps, rewards = [], [], [], [], []
    images: list[Image.Image] = []
    zero_rewards = torch.zeros(batch_size, dtype=torch.float32)
    for timestep_tensor in pipe.scheduler.timesteps:
        timestep = int(timestep_tensor.item())
        states.append(latents.detach().float().cpu())
        noise_pred = predict_noise(pipe, latents, timestep_tensor, prompt_embeds, config.guidance_scale)
        next_latents, log_prob = ddpm_step_with_log_prob(
            pipe.scheduler, noise_pred, timestep, latents, generator, eta=config.eta
        )
        actions.append(next_latents.detach().float().cpu())
        log_probs.append(log_prob.detach().float().cpu())
        timesteps.append(timestep)
        latents = next_latents
        # Compute reward on the last timestep
        if timestep_tensor == pipe.scheduler.timesteps[-1]:
            images = decode_latents(pipe, latents)
            rewards.append(reward_fn(images, prompts).detach().float().cpu())
        else:
            rewards.append(zero_rewards.clone())
    if not images:
        images = decode_latents(pipe, latents)
    pipe.unet.train()
    
    # Normalize rewards to have mean 0 and variance 1
    rewards_tensor = torch.stack(rewards, dim=1)
    normalized_rewards = normalize_rewards(rewards_tensor)
    
    return {
        "prompts": prompts,
        "states": torch.stack(states, dim=1),
        "actions": torch.stack(actions, dim=1),
        "old_log_probs": torch.stack(log_probs, dim=1),
        "timesteps": torch.tensor(timesteps, dtype=torch.long),
        "rewards": normalized_rewards,
        "images": images,
    }


def train(config: DDPOConfig) -> list[dict[str, float]]:
    output_dir = prepare_output(config)
    gpu_ids = resolve_gpu_ids(config)
    device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
    dtype = torch.float32 if device.type == "cuda" and config.mixed_precision else torch.float16
    generator = set_seed(config.seed, device)
    print(device, dtype, f"gpu_ids={gpu_ids}")
    reward_fn = make_reward_fn(device, config.reward_type)
    pipe = load_lora_pipeline(config, device=device, dtype=dtype, gpu_ids=gpu_ids)
    parameters = trainable_parameters(pipe.unet)
    print(f"Training {sum(parameter.numel() for parameter in parameters):,} LoRA parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )
    vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    history: list[dict[str, float]] = []
    for epoch in trange(1, config.train_epochs + 1):
        rollout = collect_rollouts(
            pipe, config.rollouts_per_epoch, config, device, dtype, generator, reward_fn, vae_scale_factor
        )
        
        # Save trajectory data
        save_trajectory_data("ddpo", epoch, rollout)
        
        # apply PPO update to the model using the collected rollouts and save the metrics to history
        metrics = ppo_update(pipe, rollout, optimizer, config, device, dtype)
        metrics["epoch"] = epoch
        history.append(metrics)
        save_json(output_dir / "history.json", history)
        
        # Save evaluation metrics from the current epoch training
        save_evaluation_metrics("ddpo", epoch, metrics, rollout.get("images"), rollout.get("prompts"), output_dir)
        
        if epoch % config.log_every == 0:
            print(metrics)
        if epoch % config.save_every == 0:
            save_lora_weights(pipe, output_dir / f"lora_epoch_{epoch:04d}")
            for index, image in enumerate(rollout["images"]):
                image.save(output_dir / f"epoch_{epoch:04d}_sample_{index:02d}.png")
        del rollout
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
        if config.evaluate_every > 0 and epoch % config.evaluate_every == 0:
            evaluate(pipe, config, device, epoch=epoch)
    
    # Save final training data
    save_training_data("ddpo", history)
    
    final_dir = output_dir / "lora_final"
    save_lora_weights(pipe, final_dir)
    save_json(output_dir / "config.json", asdict(config))
    save_json(output_dir / "history.json", history)
    print(f"Saved fine-tuned LoRA weights to {final_dir}")
    return history


def main() -> None:
    train(parse_config(DDPOConfig, __doc__ or "Train DDPO"))


if __name__ == "__main__":
    main()
