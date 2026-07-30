"""B2-DiffuRL with backward-progressive, branch-based sampling.

Converted from ``clover/exp/b2diffurl.ipynb``. Run with
``python -m clover.baselines.b2diffurl``.
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
    generate_eval_images,
    make_reward_fn,
    normalize_rewards,
    parse_config,
    prepare_output,
    save_evaluation_metrics,
    save_trajectory_data,
    save_training_data,
)
from clover.utils.prompts import DEFAULT_EVAL_PROMPTS, DEFAULT_TRAIN_PROMPTS
from clover.utils.baseline_utils import (
    backward_progressive_interval_length,
    decode_latents,
    ddpm_step_with_log_prob,
    encode_prompts,
    load_lora_pipeline,
    ppo_update,
    predict_noise,
    resolve_gpu_ids,
    sample_prompt_batch,
    save_image_grid_outputs,
    save_json,
    save_lora_weights,
    select_branch_extremes,
    set_seed,
    standard_eval_prompts,
    suffix_step_indices,
    trainable_parameters,
    unet_config,
)


@dataclass
class B2DiffuRLConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    output_dir: str = "outputs/b2diffurl"
    seed: int = 17
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    use_data_parallel: bool = False
    prompt: str = "a colorful clover field at sunrise, high detail"
    negative_prompt: str = "blurry, low quality, distorted"
    train_prompts: tuple[str, ...] = DEFAULT_TRAIN_PROMPTS
    eval_prompts: tuple[str, ...] = DEFAULT_EVAL_PROMPTS
    reward_type: str = "aesthetic"  # Options: "aesthetic" (more to come)
    height: int = 512
    width: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    eta: float = 1.0
    rollouts_per_epoch: int = 1
    branch_size: int = 3
    initial_interval_steps: int = 6
    train_epochs: int = 4
    ppo_epochs: int = 4
    minibatch_size: int = 1
    learning_rate: float = 1e-9
    adam_epsilon: float = 1e-4
    lora_rank: int = 2
    lora_alpha: int = 2
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("to_v",)
    clip_range: float = 1e-4
    ppo_log_ratio_clip: float = 2.0
    min_reward_gap: float = 0.0
    max_grad_norm: float = 0.1
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    log_every: int = 1
    save_every: int = 5


@torch.no_grad()
def collect_branch_rollouts(
    pipe: Any,
    batch_size: int,
    interval_steps: int,
    config: B2DiffuRLConfig,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    reward_fn: Any,
    vae_scale_factor: int,
) -> dict[str, Any]:
    if config.branch_size < 2:
        raise ValueError("branch_size must be at least 2 for branch-based sampling")
    pipe.unet.eval()
    root_prompts = sample_prompt_batch(config.train_prompts, batch_size)
    root_embeds = encode_prompts(pipe, root_prompts, config.negative_prompt, device, dtype)
    pipe.scheduler.set_timesteps(config.num_inference_steps, device=device)
    timesteps = [int(t.item()) for t in pipe.scheduler.timesteps]
    step_indices = suffix_step_indices(len(timesteps), interval_steps)
    branch_start_idx = step_indices[0]
    latent_shape = (
        batch_size,
        unet_config(pipe).in_channels,
        config.height // vae_scale_factor,
        config.width // vae_scale_factor,
    )
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
    latents = latents * pipe.scheduler.init_noise_sigma
    for prefix_idx in range(branch_start_idx):
        timestep_tensor = pipe.scheduler.timesteps[prefix_idx]
        noise_pred = predict_noise(pipe, latents, timestep_tensor, root_embeds, config.guidance_scale)
        latents, _ = ddpm_step_with_log_prob(
            pipe.scheduler,
            noise_pred,
            timesteps[prefix_idx],
            latents,
            generator,
            eta=config.eta,
        )
    branch_prompts = [prompt for prompt in root_prompts for _ in range(config.branch_size)]
    branch_embeds = encode_prompts(pipe, branch_prompts, config.negative_prompt, device, dtype)
    latents = latents.repeat_interleave(config.branch_size, dim=0)
    branch_batch = len(branch_prompts)
    states, actions, log_probs, active_timesteps = [], [], [], []
    rewards: list[Tensor] = []
    zero_rewards = torch.zeros(branch_batch, dtype=torch.float32)
    for step_idx in step_indices:
        timestep_tensor = pipe.scheduler.timesteps[step_idx]
        timestep = timesteps[step_idx]
        states.append(latents.detach().float().cpu())
        noise_pred = predict_noise(pipe, latents, timestep_tensor, branch_embeds, config.guidance_scale)
        next_latents, log_prob = ddpm_step_with_log_prob(
            pipe.scheduler, noise_pred, timestep, latents, generator, eta=config.eta
        )
        actions.append(next_latents.detach().float().cpu())
        log_probs.append(log_prob.detach().float().cpu())
        active_timesteps.append(timestep)
        latents = next_latents
        rewards.append(zero_rewards.clone())
    images = decode_latents(pipe, latents)
    terminal_rewards = reward_fn(images, branch_prompts).detach().float().cpu()
    rewards[-1] = terminal_rewards
    
    # Normalize rewards to have mean 0 and variance 1 before selection
    rewards_tensor = torch.stack(rewards, dim=1)
    normalized_rewards = normalize_rewards(rewards_tensor)
    
    selected_idx, advantages, reward_pairs = select_branch_extremes(
        terminal_rewards,
        branch_size=config.branch_size,
        min_reward_gap=config.min_reward_gap,
    )
    selected_idx = selected_idx.cpu()
    pipe.unet.train()
    return {
        "prompts": [branch_prompts[i] for i in selected_idx.tolist()],
        "states": torch.stack(states, dim=1)[selected_idx],
        "actions": torch.stack(actions, dim=1)[selected_idx],
        "old_log_probs": torch.stack(log_probs, dim=1)[selected_idx],
        "timesteps": torch.tensor(active_timesteps, dtype=torch.long),
        "rewards": normalized_rewards[selected_idx],
        "advantages": advantages.cpu(),
        "branch_reward_pairs": reward_pairs.cpu(),
        "images": [images[i] for i in selected_idx.tolist()],
        "all_images": images,
        "all_prompts": branch_prompts,
        "all_terminal_rewards": terminal_rewards,
        "interval_steps": interval_steps,
        "branch_start_idx": branch_start_idx,
    }


def train(config: B2DiffuRLConfig) -> list[dict[str, float]]:
    output_dir = prepare_output(config)
    gpu_ids = resolve_gpu_ids(config)
    device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
    dtype = torch.float16 if device.type == "cuda" and config.mixed_precision else torch.float32
    generator = set_seed(config.seed, device)
    print(device, dtype, f"gpu_ids={gpu_ids}")
    reward_fn = make_reward_fn(device, config.reward_type)
    pipe = load_lora_pipeline(config, device=device, dtype=dtype, gpu_ids=gpu_ids)
    parameters = trainable_parameters(pipe.unet)
    print(f"Training {sum(parameter.numel() for parameter in parameters):,} LoRA parameters")
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    history: list[dict[str, float]] = []
    for epoch in trange(1, config.train_epochs + 1):
        interval_steps = backward_progressive_interval_length(
            epoch, config.train_epochs, config.num_inference_steps, config.initial_interval_steps
        )
        rollout = collect_branch_rollouts(
            pipe,
            config.rollouts_per_epoch,
            interval_steps,
            config,
            device,
            dtype,
            generator,
            reward_fn,
            vae_scale_factor,
        )
        
        # Save trajectory data
        save_trajectory_data("b2diffurl", epoch, rollout)
        
        metrics = ppo_update(
            pipe,
            rollout,
            optimizer,
            config,
            device,
            dtype,
            advantages=rollout["advantages"],
            reward_values=rollout["rewards"][:, -1],
        )
        reward_pairs = rollout["branch_reward_pairs"]
        metrics.update(
            mean_reward=metrics["reward_mean"],
            mean_pair_gap=float((reward_pairs[:, 0] - reward_pairs[:, 1]).mean())
            if reward_pairs.numel()
            else float("nan"),
            epoch=epoch,
            interval_steps=interval_steps,
            branch_start_idx=rollout["branch_start_idx"],
        )
        history.append(metrics)
        
        # Save evaluation metrics
        save_evaluation_metrics("b2diffurl", epoch, metrics, rollout.get("images"), rollout.get("prompts"), output_dir)
        
        if epoch % config.log_every == 0:
            print(metrics)
        if epoch % config.save_every == 0:
            save_lora_weights(pipe, output_dir / f"lora_epoch_{epoch:04d}")
            save_image_grid_outputs(
                rollout["images"], rollout["prompts"], output_dir, f"epoch_{epoch:04d}_sample"
            )
        save_json(output_dir / "history.json", history)
        del rollout
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    prompts = standard_eval_prompts(config)
    images = generate_eval_images(pipe, prompts, config, device)
    rewards = reward_fn(images, prompts).detach().float().cpu().tolist()
    save_image_grid_outputs(images, prompts, output_dir, "eval")
    
    # Save final training data
    save_training_data("b2diffurl", history)
    
    save_json(
        output_dir / "eval_metrics.json",
        [{"prompt": prompt, "reward": reward} for prompt, reward in zip(prompts, rewards)],
    )
    final_dir = output_dir / "lora_final"
    save_lora_weights(pipe, final_dir)
    save_json(output_dir / "config.json", asdict(config))
    save_json(output_dir / "history.json", history)
    print(f"Saved fine-tuned LoRA weights to {final_dir}")
    return history


def main() -> None:
    train(parse_config(B2DiffuRLConfig, __doc__ or "Train B2-DiffuRL"))


if __name__ == "__main__":
    main()
