"""MD3PO: max diversity denoising diffusion policy optimization.

Converted from ``clover/exp/ddpo.ipynb``. Run with
``python -m clover.baselines.md3po``.
"""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
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
from clover.utils.baseline_utils import (
    ddpm_step_with_log_prob,
    decode_latents,
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
from clover.utils.rewards_utils import clip_prompt_cosine_similarity, clip_image_cosine_similarity
from clover.utils.prompts import B2_FULL_EVAL_PROMPTS, B2_FULL_TRAIN_PROMPTS


@dataclass
class MD3POConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    output_dir: str = "outputs/md3po"
    seed: int = 17
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    use_data_parallel: bool = False
    prompt: str = "a colorful clover field at sunrise, high detail"
    negative_prompt: str = "blurry, low quality, distorted"
    train_prompts: tuple[str, ...] = B2_FULL_TRAIN_PROMPTS
    eval_prompts: tuple[str, ...] = B2_FULL_EVAL_PROMPTS
    reward_type: str = "clip"
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
    config: MD3POConfig,
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

        if timestep_tensor == pipe.scheduler.timesteps[-1]:
            images = decode_latents(pipe, latents)
            rewards.append(reward_fn(images, prompts).detach().float().cpu())
        else:
            rewards.append(zero_rewards.clone())

    if not images:
        images = decode_latents(pipe, latents)
    pipe.unet.train()

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


def calculate_ssim(image_a, image_b):
    """Calculate grayscale SSIM between two RGB images."""
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray_a, gray_b, full=True)
    return score


def _to_rgba_uint8(frame_chw):
    """Convert a single frame from CHW tensor/array to HWC uint8 RGBA."""
    frame = frame_chw.detach().cpu().numpy() if hasattr(frame_chw, "detach") else np.asarray(frame_chw)

    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = np.repeat(frame, 4, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        alpha = np.ones((frame.shape[0], frame.shape[1], 1), dtype=frame.dtype)
        frame = np.concatenate([frame, alpha], axis=2)

    if np.issubdtype(frame.dtype, np.floating):
        fmin, fmax = float(frame.min()), float(frame.max())
        if fmax > fmin:
            frame = (frame - fmin) / (fmax - fmin)
        else:
            frame = np.zeros_like(frame)
        frame = (frame * 255).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame


def md3po_combined_rollouts(reference_rollout, trajectories=None, diversity_threshold=0.5, prompt_similarity_threshold=0.9):
    """Combine the reference rollout with prompt-matched diverse saved rollouts."""
    if trajectories is None:
        try:
            trajectories = torch.load("clover/data/md3po/trajectories.pt", map_location="cpu", weights_only=True)
        except FileNotFoundError:
            return reference_rollout

    def _canonicalize_rollout(rollout_data):
        if not isinstance(rollout_data, dict):
            raise TypeError("rollout_data must be a dict")

        field_aliases = {
            "state": ("state", "states"),
            "action": ("action", "actions"),
            "prompts": ("prompts",),
            "rewards": ("rewards",),
            "timesteps": ("timesteps",),
            "old_log_probs": ("old_log_probs",),
            "images": ("images",),
        }

        canonical_rollout = {}
        missing_fields = []
        for canonical_name, aliases in field_aliases.items():
            for alias in aliases:
                if alias in rollout_data:
                    canonical_rollout[canonical_name] = rollout_data[alias]
                    break
            else:
                missing_fields.append(canonical_name)

        if missing_fields:
            raise KeyError(f"Rollout is missing fields: {missing_fields}")

        return canonical_rollout

    def _to_training_rollout(rollout_data):
        return {
            "prompts": rollout_data["prompts"],
            "states": rollout_data["state"],
            "actions": rollout_data["action"],
            "old_log_probs": rollout_data["old_log_probs"],
            "timesteps": rollout_data["timesteps"],
            "rewards": rollout_data["rewards"],
            "images": rollout_data["images"],
        }

    def _extract_prompt(rollout_data):
        prompt = rollout_data.get("prompts")
        if isinstance(prompt, (list, tuple)):
            return prompt[0] if prompt else None
        return prompt

    def _extract_state_batch(rollout_data):
        state_batch = rollout_data.get("state")
        if state_batch is None:
            raise ValueError("Missing 'state' in rollout data")

        while hasattr(state_batch, "ndim") and state_batch.ndim > 4:
            if state_batch.shape[0] == 1:
                state_batch = state_batch[0]
            else:
                state_batch = state_batch[-1]

        if not hasattr(state_batch, "shape") or state_batch.ndim != 4:
            raise ValueError(
                f"Expected batch shape (N, C, H, W), got {getattr(state_batch, 'shape', type(state_batch))}"
            )
        return state_batch

    def _infer_concat_dim(values, field_name, sample_count):
        first_value = values[0]
        if not torch.is_tensor(first_value):
            raise TypeError(f"Expected tensor values for '{field_name}', got {type(first_value)}")

        preferred_dims = []
        if first_value.ndim >= 2 and first_value.shape[1] == sample_count:
            preferred_dims.append(1)
        if first_value.shape[0] == sample_count:
            preferred_dims.append(0)
        if 0 not in preferred_dims:
            preferred_dims.append(0)

        for dim in preferred_dims:
            if all(
                value.ndim == first_value.ndim
                and all(
                    value.shape[axis] == first_value.shape[axis]
                    for axis in range(first_value.ndim)
                    if axis != dim
                )
                for value in values[1:]
            ):
                return dim

        raise ValueError(
            f"Could not determine concat axis for '{field_name}' with shapes "
            f"{[tuple(value.shape) for value in values]} and sample_count={sample_count}"
        )

    def _concat_values(values, field_name, sample_count):
        first_value = values[0]
        if torch.is_tensor(first_value):
            concat_dim = _infer_concat_dim(values, field_name, sample_count)
            return torch.cat(values, dim=concat_dim)
        if isinstance(first_value, list):
            merged = []
            for value in values:
                merged.extend(value)
            return merged
        if isinstance(first_value, tuple):
            merged = []
            for value in values:
                merged.extend(list(value))
            return merged
        raise TypeError(f"Unsupported field type for '{field_name}': {type(first_value)}")

    if not isinstance(reference_rollout, dict):
        raise TypeError("reference_rollout must be a rollout data structure (dict)")

    reference_rollout = _canonicalize_rollout(reference_rollout)
    reference_prompt = _extract_prompt(reference_rollout)
    reference_batch = _extract_state_batch(reference_rollout)
    reference_sample_count = reference_batch.shape[0]
    reference_final_image = _to_rgba_uint8(reference_batch[-1])[..., :3]
    reference_final_image_pil = Image.fromarray(reference_final_image, mode="RGB")

    required_fields = ("state", "action", "prompts", "rewards", "timesteps", "old_log_probs", "images")
    rollouts_to_combine = [reference_rollout]

    for rollout in trajectories.values():
        if not isinstance(rollout, dict):
            continue

        rollout = _canonicalize_rollout(rollout)
        
        # if the prompt similarity is below the threshold, skip this rollout else use the rollout for combination
        if clip_prompt_cosine_similarity(_extract_prompt(rollout), reference_prompt) < prompt_similarity_threshold:
            continue

        rollout_batch = _extract_state_batch(rollout)
        if rollout_batch.shape[0] != reference_sample_count:
            raise ValueError(
                f"Mismatched sample count: reference has {reference_sample_count}, "
                f"rollout has {rollout_batch.shape[0]}"
            )

        rollout_final_image = _to_rgba_uint8(rollout_batch[-1])[..., :3]
        rollout_final_image_pil = Image.fromarray(rollout_final_image, mode="RGB")
        ssim_score = calculate_ssim(reference_final_image, rollout_final_image)
        clip_cosine_score = clip_image_cosine_similarity(reference_final_image_pil, rollout_final_image_pil)
        if clip_cosine_score <= diversity_threshold:
            rollouts_to_combine.append(rollout)

    if len(rollouts_to_combine) == 1:
        return _to_training_rollout(reference_rollout)

    combined_rollout = {}
    for field in required_fields:
        combined_rollout[field] = _concat_values(
            [rollout[field] for rollout in rollouts_to_combine],
            field,
            reference_sample_count,
        )

    return _to_training_rollout(combined_rollout)

def train(config: MD3POConfig) -> list[dict[str, float]]:
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
        
        combined_rollout = md3po_combined_rollouts(rollout, diversity_threshold=0.5)
        
        # apply PPO update to the model using the collected rollouts and save the metrics to history
        metrics = ppo_update(pipe, combined_rollout, optimizer, config, device, dtype)
        metrics["epoch"] = epoch
        history.append(metrics)
        save_json(output_dir / "history.json", history)
        
        # Save trajectory data for the current epoch
        save_trajectory_data("md3po", epoch, rollout)
        
        # Save evaluation metrics from the current epoch training
        save_evaluation_metrics("md3po", epoch, metrics, rollout.get("images"), rollout.get("prompts"), output_dir)
        
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
    save_training_data("md3po", history)
    
    final_dir = output_dir / "lora_final"
    save_lora_weights(pipe, final_dir)
    save_json(output_dir / "config.json", asdict(config))
    save_json(output_dir / "history.json", history)
    print(f"Saved fine-tuned LoRA weights to {final_dir}")
    return history


def main() -> None:
    train(parse_config(MD3POConfig, __doc__ or "Train MD3PO"))


if __name__ == "__main__":
    main()
