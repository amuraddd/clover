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
    is_stochastic_ddpm_transition,
    encode_prompts,
    finite_trainable_parameters,
    gaussian_kl,
    load_lora_pipeline,
    load_reference_pipeline,
    load_training_checkpoint,
    normalize_advantages,
    parameter_delta_norm,
    reward_metrics_by_prompt_category,
    predict_noise_chunked,
    resolve_gpu_ids,
    restore_trainable_parameters,
    safe_metric_mean,
    sample_prompt_batch,
    save_json,
    save_lora_weights,
    save_training_checkpoint,
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
    min_log_prob_std: float = 1e-4
    rollout_chunk_size: int = 32
    rollouts_per_epoch: int = 256
    train_epochs: int = 10
    dpok_epochs: int = 2
    minibatch_size: int = 64
    learning_rate: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("to_v", "to_k", "to_q", "to_out.0")
    reward_weight: float = 1.0
    kl_weight: float = 0.01
    clip_range: float = 0.1  # CLI compatibility placeholder; unused by DPOK.
    target_kl: float = 0.1
    normalize_rewards: bool = True
    max_grad_norm: float = 1.0
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
    states, actions, log_probs, timesteps = [], [], [], []
    images: list[Image.Image] = []
    terminal_rewards: Tensor | None = None
    for timestep_tensor in pipe.scheduler.timesteps:
        timestep = int(timestep_tensor.item())
        trainable_transition = is_stochastic_ddpm_transition(
            pipe.scheduler, timestep, eta=config.eta, min_std=config.min_log_prob_std
        )
        if trainable_transition:
            states.append(latents.detach().float().cpu())
        noise_pred = predict_noise_chunked(
            pipe,
            latents,
            timestep_tensor,
            prompt_embeds,
            config.guidance_scale,
            config.rollout_chunk_size,
        )
        next_latents, log_prob = ddpm_step_with_log_prob(
            pipe.scheduler, noise_pred, timestep, latents, generator, eta=config.eta
        )
        if trainable_transition:
            actions.append(next_latents.detach().float().cpu())
            log_probs.append(log_prob.detach().float().cpu())
            timesteps.append(timestep)
        latents = next_latents
        if timestep_tensor == pipe.scheduler.timesteps[-1]:
            images = decode_latents(pipe, latents)
            terminal_rewards = reward_fn(images, prompts).detach().float().cpu()
    if not images:
        images = decode_latents(pipe, latents)
        terminal_rewards = reward_fn(images, prompts).detach().float().cpu()
    pipe.unet.train()
    if terminal_rewards is None:
        raise RuntimeError("terminal rewards were not computed")
    rewards = terminal_rewards[:, None].expand(-1, len(timesteps)).clone()
    
    return {
        "prompts": prompts,
        "states": torch.stack(states, dim=1),
        "actions": torch.stack(actions, dim=1),
        "old_log_probs": torch.stack(log_probs, dim=1),
        "timesteps": torch.tensor(timesteps, dtype=torch.long),
        "rewards": rewards,
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
    terminal_rewards = rewards[:, 0]
    policy_weights = (
        normalize_advantages(terminal_rewards) if config.normalize_rewards else terminal_rewards
    ).to(device)
    batch_size, trajectory_len = states.shape[:2]
    indices = torch.arange(batch_size)
    losses, policy_losses, kl_losses, kls = [], [], [], []
    grad_norms, parameter_update_norms = [], []
    timestep_kls: dict[int, list[float]] = {int(timestep): [] for timestep in timesteps}
    skipped_updates = 0
    early_stopped = False
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
                minibatch_transition_kls = []
                for step_idx, timestep in enumerate(timesteps):
                    t = torch.tensor(timestep, device=device, dtype=torch.long)
                    state = states[mb_idx, step_idx].to(device=device, dtype=dtype)
                    action = actions[mb_idx, step_idx].to(device=device, dtype=dtype)
                    noise_pred = predict_noise_chunked(
                        pipe,
                        state,
                        t,
                        prompt_embeds,
                        config.guidance_scale,
                        config.rollout_chunk_size,
                    )
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
                        reference_noise = predict_noise_chunked(
                            reference_pipe,
                            state,
                            t,
                            reference_embeds,
                            config.guidance_scale,
                            config.rollout_chunk_size,
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
                    transition_kl = float(step_kl.mean().detach().cpu())
                    minibatch_transition_kls.append(transition_kl)
                    kls.append(transition_kl)
                    timestep_kls[int(timestep)].append(transition_kl)
                minibatch_kl = safe_metric_mean(minibatch_transition_kls)
                if config.target_kl > 0 and minibatch_kl > config.target_kl:
                    optimizer.zero_grad(set_to_none=True)
                    early_stopped = True
                    print(
                        f"Stopped DPOK update early: transition KL {minibatch_kl:.6f} "
                        f"exceeded target {config.target_kl:.6f}"
                    )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters(pipe.unet), config.max_grad_norm
                    )
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError("non-finite LoRA gradients")
                    grad_norms.append(float(grad_norm.detach().cpu()))
                    optimizer.step()
                    if not finite_trainable_parameters(pipe.unet):
                        restore_trainable_parameters(pipe.unet, snapshot)
                        clear_optimizer_state(optimizer)
                        raise FloatingPointError("AdamW produced non-finite LoRA parameters")
                    parameter_update_norms.append(parameter_delta_norm(pipe.unet, snapshot))
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
            if early_stopped:
                break
        if early_stopped:
            break
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
        "early_stopped": early_stopped,
        "grad_norm_pre_clip_mean": safe_metric_mean(grad_norms),
        "grad_norm_pre_clip_max": max(grad_norms, default=float("nan")),
        "parameter_update_norm_mean": safe_metric_mean(parameter_update_norms),
        "timestep_kl": {str(t): safe_metric_mean(values) for t, values in timestep_kls.items()},
        "reward_by_prompt_category": reward_metrics_by_prompt_category(prompts, terminal_rewards),
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
    last_epoch, history = load_training_checkpoint(pipe, optimizer, output_dir, device, generator)
    for epoch in trange(last_epoch + 1, config.train_epochs + 1):
        rollout = collect_rollouts(
            pipe, config.rollouts_per_epoch, config, device, dtype, generator, reward_fn, vae_scale_factor
        )
        
        # Save trajectory data
        # save_trajectory_data("dpok", epoch, rollout)
        
        metrics = dpok_update(pipe, reference_pipe, rollout, optimizer, config, device, dtype)
        metrics["epoch"] = epoch
        history.append(metrics)
        save_json(output_dir / "history.json", history)
        
        # Save evaluation metrics
        save_evaluation_metrics("dpok", epoch, metrics, rollout.get("images"), rollout.get("prompts"), output_dir)
        
        if epoch % config.log_every == 0:
            print(metrics)
        if epoch % config.save_every == 0:
            save_training_checkpoint(pipe, optimizer, output_dir, epoch, history, generator)
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
