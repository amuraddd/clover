"""DPOK: diffusion policy optimization with online KL regularization.

Converted from ``clover/exp/dpok.ipynb``. Run with
``python -m clover.baselines.dpok``.
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
    clear_optimizer_state,
    clone_trainable_parameters,
    decode_latents,
    ddpm_mean_std,
    ddpm_step_with_log_prob,
    encode_prompts,
    finite_trainable_parameters,
    gaussian_kl,
    load_lora_pipeline,
    load_reference_pipeline,
    normalize_advantages,
    predict_noise,
    resolve_gpu_ids,
    restore_trainable_parameters,
    safe_metric_mean,
    sample_prompt_batch,
    save_json,
    save_lora_weights,
    set_seed,
    trainable_parameters,
    unet_config,
)


@dataclass
class DPOKConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    output_dir: str = "outputs/dpok"
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
    dpok_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 1e-9
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 2
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("to_v", "to_k", "to_q")
    reward_weight: float = 1.0
    kl_weight: float = 0.01
    normalize_rewards: bool = False
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
    config: DPOKConfig,
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


def dpok_update(
    pipe: Any,
    reference_pipe: Any,
    rollout: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    config: DPOKConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    states, actions, rewards = rollout["states"], rollout["actions"], rollout["rewards"]
    timesteps = rollout["timesteps"].tolist()
    prompts = rollout["prompts"]
    terminal_rewards = rewards.sum(dim=1)
    policy_weights = (
        normalize_advantages(terminal_rewards) if config.normalize_rewards else terminal_rewards
    ).to(device)
    batch_size, trajectory_len = states.shape[:2]
    indices = torch.arange(batch_size)
    losses, policy_losses, kl_losses, kls = [], [], [], []
    skipped_updates = 0
    pipe.unet.train()
    reference_pipe.unet.eval()
    if not finite_trainable_parameters(pipe.unet):
        raise FloatingPointError("LoRA parameters are non-finite before the DPOK update")
    for _ in range(config.dpok_epochs):
        permutation = indices[torch.randperm(batch_size)]
        for start in range(0, batch_size, config.minibatch_size):
            mb_idx = permutation[start : start + config.minibatch_size]
            mb_prompts = [prompts[i] for i in mb_idx.tolist()]
            prompt_embeds = encode_prompts(pipe, mb_prompts, config.negative_prompt, device, dtype)
            reference_embeds = encode_prompts(
                reference_pipe, mb_prompts, config.negative_prompt, device, dtype
            )
            weights = policy_weights[mb_idx].to(device=device, dtype=torch.float32)
            snapshot = clone_trainable_parameters(pipe.unet)
            optimizer.zero_grad(set_to_none=True)
            try:
                minibatch_loss = torch.zeros((), device=device, dtype=torch.float32)
                minibatch_policy_loss = torch.zeros((), device=device, dtype=torch.float32)
                minibatch_kl_loss = torch.zeros((), device=device, dtype=torch.float32)
                for step_idx, timestep in enumerate(timesteps):
                    t = torch.tensor(timestep, device=device, dtype=torch.long)
                    state = states[mb_idx, step_idx].to(device=device, dtype=dtype)
                    action = actions[mb_idx, step_idx].to(device=device, dtype=dtype)
                    noise_pred = predict_noise(pipe, state, t, prompt_embeds, config.guidance_scale)
                    _, log_prob = ddpm_step_with_log_prob(
                        pipe.scheduler,
                        noise_pred,
                        timestep,
                        state,
                        prev_sample=action,
                        eta=config.eta,
                    )
                    current_mean, current_std = ddpm_mean_std(
                        pipe.scheduler, noise_pred, timestep, state, eta=config.eta
                    )
                    with torch.no_grad():
                        reference_noise = predict_noise(
                            reference_pipe, state, t, reference_embeds, config.guidance_scale
                        )
                        reference_mean, reference_std = ddpm_mean_std(
                            reference_pipe.scheduler,
                            reference_noise,
                            timestep,
                            state,
                            eta=config.eta,
                        )
                    step_kl = gaussian_kl(
                        current_mean, current_std, reference_mean, reference_std
                    )
                    policy_loss = -(
                        config.reward_weight * weights * log_prob.float()
                    ).mean() / trajectory_len
                    kl_loss = config.kl_weight * step_kl.mean() / trajectory_len
                    step_loss = policy_loss + kl_loss
                    if not torch.isfinite(step_loss):
                        raise FloatingPointError("non-finite DPOK loss")
                    step_loss.backward()
                    minibatch_loss += step_loss.detach()
                    minibatch_policy_loss += policy_loss.detach()
                    minibatch_kl_loss += kl_loss.detach()
                    kls.append(float(step_kl.mean().detach().cpu()))
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(pipe.unet), config.max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("non-finite LoRA gradients")
                optimizer.step()
                if not finite_trainable_parameters(pipe.unet):
                    restore_trainable_parameters(pipe.unet, snapshot)
                    clear_optimizer_state(optimizer)
                    raise FloatingPointError("AdamW produced non-finite LoRA parameters")
                losses.append(float(minibatch_loss.cpu()))
                policy_losses.append(float(minibatch_policy_loss.cpu()))
                kl_losses.append(float(minibatch_kl_loss.cpu()))
            except FloatingPointError as exc:
                restore_trainable_parameters(pipe.unet, snapshot)
                optimizer.zero_grad(set_to_none=True)
                skipped_updates += 1
                print(f"Skipped DPOK minibatch: {exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {
        "loss": safe_metric_mean(losses),
        "policy_loss": safe_metric_mean(policy_losses),
        "kl_loss": safe_metric_mean(kl_losses),
        "transition_kl": safe_metric_mean(kls),
        "reward_mean": float(terminal_rewards.mean()),
        "reward_std": float(terminal_rewards.std(unbiased=False))
        if terminal_rewards.numel() > 1
        else 0.0,
        "skipped_updates": skipped_updates,
    }


def train(config: DPOKConfig) -> list[dict[str, float]]:
    output_dir = prepare_output(config)
    gpu_ids = resolve_gpu_ids(config)
    device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
    dtype = torch.float32 if device.type == "cuda" and config.mixed_precision else torch.float16
    generator = set_seed(config.seed, device)
    print(device, dtype, f"gpu_ids={gpu_ids}")
    reward_fn = make_reward_fn(device, config.reward_type)
    pipe = load_lora_pipeline(config, device=device, dtype=dtype, gpu_ids=gpu_ids)
    reference_pipe = load_reference_pipeline(config, device=device, dtype=dtype)
    reference_pipe.scheduler.set_timesteps(config.num_inference_steps, device=device)
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
        save_trajectory_data("dpok", epoch, rollout)
        
        metrics = dpok_update(pipe, reference_pipe, rollout, optimizer, config, device, dtype)
        metrics["epoch"] = epoch
        history.append(metrics)
        save_json(output_dir / "history.json", history)
        
        # Save evaluation metrics
        save_evaluation_metrics("dpok", epoch, metrics, rollout.get("images"), rollout.get("prompts"), output_dir)
        
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
    save_training_data("dpok", history)
    
    final_dir = output_dir / "lora_final"
    save_lora_weights(pipe, final_dir)
    save_json(output_dir / "config.json", asdict(config))
    save_json(output_dir / "history.json", history)
    print(f"Saved fine-tuned LoRA weights to {final_dir}")
    return history


def main() -> None:
    train(parse_config(DPOKConfig, __doc__ or "Train DPOK"))


if __name__ == "__main__":
    main()
