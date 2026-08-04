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
from clover.utils.rewards_utils import clip_prompt_embeddings, load_clip_text_encoder
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
    """Combine rollout samples along the leading batch dimension.

    Prompt and complete-state encodings are normalized before aligned dot
    products are used for sample-wise prompt-similarity and diversity filters.
    """
    if trajectories is None:
        try:
            trajectories = torch.load("clover/data/md3po/trajectories.pt", map_location="cpu", weights_only=True)
        except FileNotFoundError:
            return reference_rollout

    def canonicalize(data):
        if not isinstance(data, dict):
            raise TypeError("rollout data must be a dict")
        aliases = {
            "state": ("state", "states"), "action": ("action", "actions"),
            "prompts": ("prompts",), "rewards": ("rewards",),
            "timesteps": ("timesteps",), "old_log_probs": ("old_log_probs",),
            "images": ("images",),
        }
        result = {}
        for name, candidates in aliases.items():
            result[name] = next((data[key] for key in candidates if key in data), None)
            if result[name] is None:
                raise KeyError(f"Rollout is missing field: {name}")
        return result

    def validate(data):
        states = data["state"]
        if not torch.is_tensor(states) or states.ndim < 2:
            raise ValueError(f"Expected batched states, got {getattr(states, 'shape', type(states))}")
        if not isinstance(data["prompts"], (list, tuple)) or len(data["prompts"]) != states.shape[0]:
            raise ValueError(f"Expected one prompt per state sample ({states.shape[0]})")
        return states

    def encode_states(states, batch_size=4):
        encoded_batches = []
        for start in range(0, states.shape[0], batch_size):
            state_batch = states[start:start + batch_size]
            flattened = state_batch.detach().to(device="cpu", dtype=torch.float32).flatten(start_dim=1)
            encoded_batches.append(torch.nn.functional.normalize(flattened, dim=-1))
        return torch.cat(encoded_batches, dim=0)

    def select_batch(data, mask):
        indices = mask.nonzero(as_tuple=False).flatten()
        selected = {}
        for field, value in data.items():
            if field == "timesteps":
                selected[field] = value
            elif torch.is_tensor(value):
                if value.ndim == 0 or value.shape[0] != mask.numel():
                    raise ValueError(f"Field '{field}' does not have batch size {mask.numel()}")
                selected[field] = value.index_select(0, indices.to(value.device))
            elif isinstance(value, (list, tuple)):
                selected[field] = [value[index] for index in indices.tolist()]
            else:
                raise TypeError(f"Unsupported field type for '{field}': {type(value)}")
        return selected

    def training_format(data):
        return {
            "prompts": data["prompts"], "states": data["state"], "actions": data["action"],
            "old_log_probs": data["old_log_probs"], "timesteps": data["timesteps"],
            "rewards": data["rewards"], "images": data["images"],
        }

    if not isinstance(reference_rollout, dict):
        raise TypeError("reference_rollout must be a dict")
    reference = canonicalize(reference_rollout)
    reference_states = validate(reference)
    reference_state_encodings = encode_states(reference_states)

    clip_model, clip_tokenizer, clip_device = load_clip_text_encoder()

    def encode_prompts(prompts):
        return clip_prompt_embeddings(
            prompts,
            model=clip_model,
            tokenizer=clip_tokenizer,
            device=clip_device,
            batch_size=4,
        )

    reference_prompt_encodings = encode_prompts(reference["prompts"])

    accepted = [reference]
    for saved_rollout in trajectories.values():
        if not isinstance(saved_rollout, dict):
            continue
        rollout = canonicalize(saved_rollout)
        states = validate(rollout)
        if states.shape != reference_states.shape:
            raise ValueError(
                f"State shape mismatch: reference has {tuple(reference_states.shape)}, "
                f"rollout has {tuple(states.shape)}"
            )
        state_scores = torch.sum(reference_state_encodings * encode_states(states), dim=-1)
        prompt_scores = torch.sum(reference_prompt_encodings * encode_prompts(rollout["prompts"]), dim=-1)
        keep = (state_scores <= diversity_threshold) & (prompt_scores >= prompt_similarity_threshold)
        if keep.any():
            accepted.append(select_batch(rollout, keep))

    if len(accepted) == 1:
        return training_format(reference)

    combined = {}
    for field in ("state", "action", "prompts", "rewards", "timesteps", "old_log_probs", "images"):
        values = [rollout[field] for rollout in accepted]
        if field == "timesteps":
            if not all(torch.equal(values[0], value) for value in values[1:]):
                raise ValueError("Cannot combine rollouts with different timestep schedules")
            combined[field] = values[0]
        elif torch.is_tensor(values[0]):
            combined[field] = torch.cat(values, dim=0)
        else:
            combined[field] = [item for value in values for item in value]
    return training_format(combined)

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
