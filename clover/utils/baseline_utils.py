from __future__ import annotations

import math
import random
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch import Tensor
from PIL import Image


def resolve_gpu_ids(config: Any) -> list[int]:
    if not torch.cuda.is_available():
        return []
    visible_count = torch.cuda.device_count()
    return [gpu_id for gpu_id in config.gpu_ids if 0 <= gpu_id < visible_count]


def set_seed(seed: int, device: torch.device) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator_device = device.type if device.type == "cpu" else str(device)
    return torch.Generator(device=generator_device).manual_seed(seed)


def unwrap_unet(unet: torch.nn.Module) -> torch.nn.Module:
    return unet.module if isinstance(unet, torch.nn.DataParallel) else unet


def unet_config(pipe: StableDiffusionPipeline):
    return unwrap_unet(pipe.unet).config


def trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def save_lora_weights(pipe: StableDiffusionPipeline, save_dir: Path | str) -> None:
    unet = unwrap_unet(pipe.unet)
    pipe.save_lora_weights(
        save_directory=save_dir,
        unet_lora_layers=get_peft_model_state_dict(unet),
        safe_serialization=True,
    )


def save_training_checkpoint(
    pipe: StableDiffusionPipeline,
    optimizer: torch.optim.Optimizer,
    output_dir: Path | str,
    epoch: int,
    history: list[dict[str, float]],
    generator: torch.Generator,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> Path:
    """Atomically overwrite the resumable checkpoint for one baseline."""
    checkpoint_dir = Path(output_dir) / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    temporary_path = checkpoint_dir / "checkpoint.pt.tmp"
    payload = {
        "epoch": epoch,
        "history": history,
        "lora_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in get_peft_model_state_dict(unwrap_unet(pipe.unet)).items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generator_state": generator.get_state(),
    }
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Saved checkpoint for completed epoch {epoch} to {checkpoint_path}")
    return checkpoint_path


def load_training_checkpoint(
    pipe: StableDiffusionPipeline,
    optimizer: torch.optim.Optimizer,
    output_dir: Path | str,
    device: torch.device,
    generator: torch.Generator,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[int, list[dict[str, float]]]:
    """Restore a baseline checkpoint and return its last epoch and history."""
    checkpoint_path = Path(output_dir) / "checkpoint" / "checkpoint.pt"
    if not checkpoint_path.is_file():
        return 0, []

    # RNG states must remain CPU ByteTensors while they are restored. Loading the
    # whole checkpoint directly onto CUDA turns them into CUDA tensors, which
    # torch.cuda.set_rng_state_all rejects.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(unwrap_unet(pipe.unet), checkpoint["lora_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    cuda_rng_state = checkpoint.get("cuda_rng_state")
    if device.type == "cuda" and cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)
    generator.set_state(checkpoint["generator_state"].cpu())
    epoch = int(checkpoint["epoch"])
    history = list(checkpoint.get("history", []))
    print(f"Resuming from checkpoint {checkpoint_path} after completed epoch {epoch}")
    return epoch, history


def load_lora_pipeline(
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
    gpu_ids: list[int] | None = None,
) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config, clip_sample=False)
    pipe.scheduler.register_to_config(clip_sample=False)
    pipe = pipe.to(device)

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=list(config.lora_target_modules),
    )
    pipe.unet.add_adapter(lora_config)
    pipe.unet.train()

    if config.gradient_checkpointing and hasattr(pipe.unet, "enable_gradient_checkpointing"):
        pipe.unet.enable_gradient_checkpointing()

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers memory efficient attention enabled")
    except Exception as exc:
        print(f"xFormers not enabled: {exc}")

    gpu_ids = gpu_ids or []
    if config.use_data_parallel and len(gpu_ids) > 1:
        pipe.unet = torch.nn.DataParallel(pipe.unet, device_ids=gpu_ids, output_device=gpu_ids[0])
        print(f"UNet wrapped with DataParallel on GPUs {gpu_ids}")

    return pipe


def load_reference_pipeline(config: Any, device: torch.device, dtype: torch.dtype) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config, clip_sample=False)
    pipe.scheduler.register_to_config(clip_sample=False)
    pipe = pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.unet.eval()
    return pipe


def encode_prompts(
    pipe: StableDiffusionPipeline,
    prompts: list[str],
    negative_prompt: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    tokenized = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    negative = pipe.tokenizer(
        [negative_prompt] * len(prompts),
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeds = pipe.text_encoder(tokenized.input_ids.to(device))[0]
        negative_embeds = pipe.text_encoder(negative.input_ids.to(device))[0]
    return torch.cat([negative_embeds, text_embeds], dim=0).to(dtype=dtype)


def predict_noise(
    pipe: StableDiffusionPipeline,
    latents: Tensor,
    timestep: Tensor | int,
    prompt_embeds: Tensor,
    guidance_scale: float,
) -> Tensor:
    latent_model_input = torch.cat([latents, latents], dim=0)
    if isinstance(pipe.unet, torch.nn.DataParallel):
        timestep = torch.as_tensor(timestep, device=latents.device).expand(latent_model_input.shape[0])
    noise_pred = pipe.unet(latent_model_input, timestep, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
    if not torch.isfinite(noise_pred).all():
        raise FloatingPointError("UNet predicted non-finite noise; skip this update or lower the learning rate.")
    noise_uncond, noise_text = noise_pred.chunk(2)
    guided_noise = noise_uncond + guidance_scale * (noise_text - noise_uncond)
    if not torch.isfinite(guided_noise).all():
        raise FloatingPointError("Classifier-free guidance produced non-finite noise; lower guidance_scale or learning_rate.")
    return guided_noise


def predict_noise_chunked(
    pipe: StableDiffusionPipeline,
    latents: Tensor,
    timestep: Tensor | int,
    prompt_embeds: Tensor,
    guidance_scale: float,
    chunk_size: int,
) -> Tensor:
    """Run classifier-free-guidance UNet inference in bounded sample chunks.

    Prompt embeddings are stored as all unconditional embeddings followed by all
    conditional embeddings. Each chunk rebuilds that ordering before delegating
    to :func:`predict_noise`, preserving output order and autograd graphs.
    """
    if chunk_size < 1:
        raise ValueError("rollout_chunk_size must be at least 1")
    batch_size = latents.shape[0]
    if batch_size < 1:
        raise ValueError("latents must contain at least one sample")
    if prompt_embeds.shape[0] != 2 * batch_size:
        raise ValueError("prompt_embeds must contain unconditional and conditional embeddings")

    negative_embeds, positive_embeds = prompt_embeds.chunk(2, dim=0)
    predictions = []
    for start in range(0, batch_size, chunk_size):
        stop = min(start + chunk_size, batch_size)
        chunk_embeds = torch.cat(
            [negative_embeds[start:stop], positive_embeds[start:stop]], dim=0
        )
        predictions.append(
            predict_noise(
                pipe, latents[start:stop], timestep, chunk_embeds, guidance_scale
            )
        )
    return torch.cat(predictions, dim=0)


def decode_latents(
    pipe: StableDiffusionPipeline,
    latents: Tensor,
    decode_batch_size: int = 4,
) -> list[Image.Image]:
    """Decode latent samples with the VAE in memory-bounded batches."""
    if decode_batch_size <= 0:
        raise ValueError("decode_batch_size must be positive.")
    if not torch.isfinite(latents).all():
        raise FloatingPointError("Cannot decode non-finite latents; the rollout/update became unstable.")

    scaled_latents = latents / pipe.vae.config.scaling_factor
    pil_images: list[Image.Image] = []
    for latent_batch in scaled_latents.split(decode_batch_size):
        with torch.no_grad():
            image_batch = pipe.vae.decode(latent_batch.to(dtype=pipe.vae.dtype)).sample
        if not torch.isfinite(image_batch).all():
            raise FloatingPointError("VAE decoded non-finite images; the latents are unstable.")
        image_batch = (image_batch / 2 + 0.5).clamp(0, 1)
        image_batch = image_batch.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        image_batch = (image_batch * 255).round().astype("uint8")
        pil_images.extend(Image.fromarray(image) for image in image_batch)
    return pil_images


def previous_timestep(scheduler: DDPMScheduler, timestep: int) -> int:
    if hasattr(scheduler, "previous_timestep"):
        prev = scheduler.previous_timestep(timestep)
        return int(prev.item() if torch.is_tensor(prev) else prev)
    step = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    return int(timestep) - step


def ddpm_transition_variance(
    scheduler: DDPMScheduler,
    timestep: int,
    device: torch.device | None = None,
) -> Tensor:
    """Return the unclamped fixed DDPM posterior variance for one transition."""
    t = int(timestep)
    prev_t = previous_timestep(scheduler, t)
    alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
    alpha_prod_t = alphas_cumprod[t]
    if prev_t >= 0:
        alpha_prod_t_prev = alphas_cumprod[prev_t]
    elif hasattr(scheduler, "final_alpha_cumprod"):
        alpha_prod_t_prev = scheduler.final_alpha_cumprod.to(
            device=alphas_cumprod.device, dtype=torch.float32
        )
    elif hasattr(scheduler, "one"):
        alpha_prod_t_prev = scheduler.one.to(
            device=alphas_cumprod.device, dtype=torch.float32
        )
    else:
        alpha_prod_t_prev = torch.ones((), device=alphas_cumprod.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
    return ((beta_prod_t_prev / beta_prod_t) * current_beta_t).clamp_min(0)


def is_stochastic_ddpm_transition(
    scheduler: DDPMScheduler,
    timestep: int,
    eta: float = 1.0,
    min_std: float = 1e-4,
) -> bool:
    """Whether a transition has enough variance for a stable log-probability."""
    if eta <= 0:
        return False
    variance = ddpm_transition_variance(scheduler, timestep)
    std = float((variance.sqrt() * eta).detach().cpu())
    return math.isfinite(std) and std >= min_std


def ddpm_mean_std(
    scheduler: DDPMScheduler,
    model_output: Tensor,
    timestep: int,
    sample: Tensor,
    eta: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Mean and std of p_theta(x_{t-1} | x_t) for fixed-variance DDPM transitions."""
    if model_output.shape[1] == sample.shape[1] * 2:
        model_output, _ = torch.split(model_output, sample.shape[1], dim=1)

    sample_float = sample.float()
    if not torch.isfinite(sample_float).all():
        raise FloatingPointError("Non-finite DDPM sample; skip this update or lower the learning rate.")
    model_output = model_output.float()
    if not torch.isfinite(model_output).all():
        raise FloatingPointError("Non-finite DDPM model output; skip this update or lower the learning rate.")
    t = int(timestep)
    prev_t = previous_timestep(scheduler, t)
    alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=torch.float32)
    alpha_prod_t = alphas_cumprod[t]
    if prev_t >= 0:
        alpha_prod_t_prev = alphas_cumprod[prev_t]
    elif hasattr(scheduler, "final_alpha_cumprod"):
        alpha_prod_t_prev = scheduler.final_alpha_cumprod.to(device=sample.device, dtype=torch.float32)
    elif hasattr(scheduler, "one"):
        alpha_prod_t_prev = scheduler.one.to(device=sample.device, dtype=torch.float32)
    else:
        alpha_prod_t_prev = torch.ones((), device=sample.device, dtype=torch.float32)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        pred_original_sample = (sample_float - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    elif prediction_type == "sample":
        pred_original_sample = model_output
    elif prediction_type == "v_prediction":
        pred_original_sample = alpha_prod_t.sqrt() * sample_float - beta_prod_t.sqrt() * model_output
    else:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")

    if scheduler.config.clip_sample:
        clip_range = getattr(scheduler.config, "clip_sample_range", 1.0)
        pred_original_sample = pred_original_sample.clamp(-clip_range, clip_range)

    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t
    pred_original_sample_coeff = alpha_prod_t_prev.sqrt() * current_beta_t / beta_prod_t
    current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t
    mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample_float

    if not torch.isfinite(mean).all():
        raise FloatingPointError("Non-finite DDPM transition mean; reduce LR/clip range or disable mixed precision.")

    variance = ddpm_transition_variance(scheduler, t, device=sample.device)
    std = eta * variance.sqrt()
    if not torch.isfinite(std).all():
        raise FloatingPointError("Non-finite DDPM transition std; reduce LR/clip range or disable mixed precision.")
    return mean, std


def transition_log_prob(
    mean: Tensor,
    std: Tensor,
    prev_sample: Tensor,
    likelihood_scale: float = 1.0,
) -> Tensor:
    """Return a scaled mean log-likelihood for one diffusion transition."""
    if likelihood_scale <= 0:
        raise ValueError("likelihood_scale must be positive")
    prev_sample = prev_sample.float()
    mean = mean.float()
    std = std.float()
    if (std <= 0).any():
        raise ValueError("A deterministic DDPM transition has no Gaussian log-probability.")
    log_prob = -0.5 * ((prev_sample - mean) / std).pow(2) - torch.log(std) - 0.5 * math.log(2 * math.pi)
    if not torch.isfinite(log_prob).all():
        raise FloatingPointError("Non-finite transition log-prob; reduce LR/clip range or disable mixed precision.")
    return log_prob.flatten(1).mean(dim=1) * likelihood_scale


def ddpm_step_with_log_prob(
    scheduler: DDPMScheduler,
    model_output: Tensor,
    timestep: int,
    sample: Tensor,
    generator: torch.Generator | None = None,
    prev_sample: Tensor | None = None,
    eta: float = 1.0,
    likelihood_scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    mean, std = ddpm_mean_std(scheduler, model_output, timestep, sample, eta=eta)
    deterministic = bool((std <= 0).all().item())
    if prev_sample is None:
        noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=torch.float32)
        prev_sample = (mean + std * noise).to(sample.dtype)
        if deterministic:
            return prev_sample, torch.zeros(sample.shape[0], device=sample.device, dtype=torch.float32)
    elif deterministic:
        raise ValueError("Cannot evaluate a log-probability for a deterministic DDPM transition.")
    log_prob = transition_log_prob(mean, std, prev_sample, likelihood_scale=likelihood_scale)
    return prev_sample, log_prob


def approximate_kl_from_log_ratio(log_ratio: Tensor) -> Tensor:
    """Estimate reverse KL stably from policy log-probability ratios."""
    log_ratio_64 = log_ratio.double()
    return (torch.expm1(log_ratio_64) - log_ratio_64).mean()


def gaussian_kl(mean: Tensor, std: Tensor, ref_mean: Tensor, ref_std: Tensor) -> Tensor:
    mean = mean.float()
    std = std.float().clamp_min(1e-6)
    ref_mean = ref_mean.float()
    ref_std = ref_std.float().clamp_min(1e-6)
    kl = torch.log(ref_std / std) + (std.pow(2) + (mean - ref_mean).pow(2)) / (2 * ref_std.pow(2)) - 0.5
    if not torch.isfinite(kl).all():
        raise FloatingPointError("Non-finite Gaussian KL; reduce LR or disable mixed precision.")
    return kl.flatten(1).mean(dim=1)


def parameter_delta_norm(module: torch.nn.Module, snapshot: list[Tensor]) -> float:
    """Return the L2 norm of trainable parameter changes from ``snapshot``."""
    squared_norm = 0.0
    with torch.no_grad():
        for parameter, saved in zip(trainable_parameters(module), snapshot):
            squared_norm += float((parameter.detach() - saved).float().pow(2).sum().cpu())
    return math.sqrt(squared_norm)


def reward_metrics_by_prompt_category(
    prompts: list[str], rewards: Tensor
) -> dict[str, dict[str, float | int]]:
    """Aggregate terminal rewards into coarse color, spatial, and action groups."""
    terminal = rewards[:, 0] if rewards.ndim > 1 else rewards
    grouped: dict[str, list[float]] = {}
    colors = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown")
    actions = ("playing", "riding", "washing")
    spatial = (" on ", " under ", "left of", "right of")
    for prompt, reward in zip(prompts, terminal.detach().float().cpu().tolist()):
        lowered = f" {prompt.lower()} "
        if any(f" {color} " in lowered for color in colors):
            category = "color"
        elif any(token in lowered for token in spatial):
            category = "spatial"
        elif any(f" {action} " in lowered for action in actions):
            category = "action"
        else:
            category = "other"
        grouped.setdefault(category, []).append(float(reward))
    return {
        category: {"mean": float(np.mean(values)), "std": float(np.std(values)), "count": len(values)}
        for category, values in sorted(grouped.items())
    }


def sample_prompt_batch(train_prompts: tuple[str, ...], batch_size: int) -> list[str]:
    """Sample a prompt batch from the configured training prompt pool.

    Args:
        train_prompts: Candidate prompts used by the experiment.
        batch_size: Number of prompts to return.

    Returns:
        A list of sampled prompt strings with length ``batch_size``.
    """
    return [random.choice(train_prompts) for _ in range(batch_size)]


def normalize_advantages(rewards: Tensor) -> Tensor:
    """Center and scale rewards into policy-gradient advantages.

    Args:
        rewards: Reward tensor with any shape.

    Returns:
        A tensor with the same shape as ``rewards`` and approximately zero
        mean/unit variance when more than one value is present.
    """
    advantages = rewards - rewards.mean()
    if rewards.numel() > 1:
        advantages = advantages / (rewards.std(unbiased=False) + 1e-8)
    return advantages


def finite_trainable_parameters(module: torch.nn.Module) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in trainable_parameters(module))


def clone_trainable_parameters(module: torch.nn.Module) -> list[Tensor]:
    return [parameter.detach().clone() for parameter in trainable_parameters(module)]


def restore_trainable_parameters(module: torch.nn.Module, snapshot: list[Tensor]) -> None:
    with torch.no_grad():
        for parameter, saved in zip(trainable_parameters(module), snapshot):
            parameter.copy_(saved)


def clear_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    optimizer.state.clear()


def safe_metric_mean(values: list[float]) -> float:
    """Return a finite metric mean or ``nan`` when no values were recorded.

    Args:
        values: Metric samples collected during training.

    Returns:
        The arithmetic mean as a float, or ``nan`` for an empty list.
    """
    return float(np.mean(values)) if values else float("nan")


def ppo_update(
    pipe: StableDiffusionPipeline,
    rollout: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
    advantages: Tensor | None = None,
    reward_values: Tensor | None = None,
) -> dict[str, float]:
    """Run a clipped PPO-style update over diffusion transition log-probs.

    This shared update is used by DDPO and B2-DiffuRL. DDPO passes dense or
    terminal rollout rewards that are normalized into per-step advantages.
    B2-DiffuRL passes signed best/worst branch-pair advantages while reusing the
    same clipped probability-ratio objective.

    Args:
        pipe: Stable Diffusion pipeline whose LoRA UNet is being optimized.
        rollout: Dictionary containing ``states``, ``actions``,
            ``old_log_probs``, ``timesteps``, and ``prompts``. State/action
            tensors are expected to have shape ``[batch, steps, ...]``.
        optimizer: Optimizer for trainable UNet parameters.
        config: Baseline configuration with PPO hyperparameters, prompt
            settings, guidance scale, eta, and gradient clipping values.
        device: Device used for model execution.
        dtype: Floating-point dtype used for latent/model tensors.
        advantages: Optional precomputed advantages. Shape may be ``[batch]``
            for one advantage per trajectory or ``[batch, steps]`` for per-step
            advantages. When omitted, normalized rollout rewards are used.
        reward_values: Optional tensor used only for reward summary metrics.
            Defaults to ``rollout["rewards"]`` when present.

    Returns:
        A dictionary with PPO loss, approximate KL, clipping fraction, reward
        summaries, skipped update count, and selected sample count.
    """
    states = rollout["states"]
    actions = rollout["actions"]
    old_log_probs = rollout["old_log_probs"]
    timesteps = rollout["timesteps"].tolist()
    prompts = rollout["prompts"]
    rewards = rollout.get("rewards")
    if advantages is None:
        if rewards is None:
            raise ValueError("advantages must be provided when rollout does not include rewards.")
        advantages = normalize_advantages(rewards)
    if reward_values is None and torch.is_tensor(rewards):
        reward_values = rewards

    batch_size = old_log_probs.shape[0]
    if batch_size == 0:
        return {
            "loss": float("nan"),
            "approx_kl": float("nan"),
            "clip_frac": float("nan"),
            "reward_mean": float("nan"),
            "reward_std": float("nan"),
            "raw_reward_mean": float("nan"),
            "raw_reward_std": float("nan"),
            "advantage_mean": float("nan"),
            "advantage_std": float("nan"),
            "skipped_updates": 0,
            "early_stopped": False,
            "selected_samples": 0,
        }

    trajectory_len = old_log_probs.shape[1]
    indices = torch.arange(batch_size)
    losses, approx_kls, clip_fracs = [], [], []
    grad_norms, parameter_update_norms, log_ratio_values = [], [], []
    timestep_kls: dict[int, list[float]] = {int(timestep): [] for timestep in timesteps}
    timestep_clip_fracs: dict[int, list[float]] = {int(timestep): [] for timestep in timesteps}
    skipped_updates = 0
    early_stopped = False

    pipe.unet.train()
    if not finite_trainable_parameters(pipe.unet):
        raise FloatingPointError("LoRA parameters are already non-finite; reload the pipeline before continuing.")

    for _ in range(config.ppo_epochs):
        permutation = indices[torch.randperm(batch_size)]
        for start in range(0, batch_size, config.minibatch_size):
            mb_idx = permutation[start:start + config.minibatch_size]
            mb_prompts = [prompts[i] for i in mb_idx.tolist()]
            prompt_embeds = encode_prompts(pipe, mb_prompts, config.negative_prompt, device, dtype)
            mb_advantages = advantages[mb_idx]
            parameter_snapshot = clone_trainable_parameters(pipe.unet)

            optimizer.zero_grad(set_to_none=True)
            try:
                minibatch_kls = []
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
                        likelihood_scale=getattr(config, "likelihood_scale", 1.0),
                    )

                    old = old_log_probs[mb_idx, step_idx].to(device=device, dtype=log_prob.dtype)
                    if mb_advantages.ndim == 1:
                        adv = mb_advantages.to(device=device, dtype=log_prob.dtype)
                    else:
                        adv = mb_advantages[:, step_idx].to(device=device, dtype=log_prob.dtype)

                    log_ratio = log_prob - old
                    ratio = torch.exp(log_ratio)
                    if not torch.isfinite(ratio).all():
                        raise FloatingPointError(
                            "Non-finite PPO probability ratio; lower the learning rate or inspect transition variance."
                        )
                    unclipped = ratio * adv
                    clipped = torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range) * adv
                    step_loss = -torch.min(unclipped, clipped).mean() / trajectory_len
                    if not torch.isfinite(step_loss):
                        raise FloatingPointError("Non-finite PPO loss; skipped this minibatch update.")
                    step_loss.backward()

                    with torch.no_grad():
                        approx_kl = approximate_kl_from_log_ratio(log_ratio)
                        clip_frac = ((ratio - 1).abs() > config.clip_range).float().mean()
                        log_ratio_values.append(log_ratio.detach().float().cpu().flatten())
                    minibatch_kls.append(float(approx_kl.detach().cpu()))
                    losses.append(float((step_loss * trajectory_len).detach().cpu()))
                    approx_kls.append(float(approx_kl.detach().cpu()))
                    clip_fracs.append(float(clip_frac.detach().cpu()))
                    timestep_kls[int(timestep)].append(float(approx_kl.detach().cpu()))
                    timestep_clip_fracs[int(timestep)].append(float(clip_frac.detach().cpu()))

                    del state, action, noise_pred, log_prob, old, adv, log_ratio, ratio, unclipped, clipped, step_loss

                minibatch_kl = safe_metric_mean(minibatch_kls)
                target_kl = getattr(config, "target_kl", None)
                if target_kl is not None and target_kl > 0 and minibatch_kl > target_kl:
                    optimizer.zero_grad(set_to_none=True)
                    early_stopped = True
                    print(
                        f"Stopped PPO update early: minibatch KL {minibatch_kl:.6f} "
                        f"exceeded target {target_kl:.6f}"
                    )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters(pipe.unet), config.max_grad_norm
                    )
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError("Non-finite LoRA gradients; skipped this minibatch update.")

                    grad_norms.append(float(grad_norm.detach().cpu()))
                    optimizer.step()
                    if not finite_trainable_parameters(pipe.unet):
                        restore_trainable_parameters(pipe.unet, parameter_snapshot)
                        clear_optimizer_state(optimizer)
                        raise FloatingPointError("AdamW produced non-finite LoRA parameters; restored previous weights and cleared optimizer state.")
                    parameter_update_norms.append(parameter_delta_norm(pipe.unet, parameter_snapshot))

            except FloatingPointError as exc:
                restore_trainable_parameters(pipe.unet, parameter_snapshot)
                clear_optimizer_state(optimizer)
                optimizer.zero_grad(set_to_none=True)
                skipped_updates += 1
                print(f"Skipped PPO minibatch: {exc}")
            finally:
                del parameter_snapshot

            if device.type == "cuda":
                torch.cuda.empty_cache()
            if early_stopped:
                break
        if early_stopped:
            break

    if torch.is_tensor(reward_values) and reward_values.numel() > 0:
        reward_values = reward_values.float()
        reward_mean = float(reward_values.mean().item())
        reward_std = float(reward_values.std(unbiased=False).item()) if reward_values.numel() > 1 else 0.0
    else:
        reward_mean = float("nan")
        reward_std = float("nan")
    advantage_values = advantages.float()
    if log_ratio_values:
        all_log_ratios = torch.cat(log_ratio_values)
        quantiles = torch.quantile(
            all_log_ratios, torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0])
        ).tolist()
        log_ratio_percentiles = dict(zip(("p0", "p1", "p50", "p99", "p100"), quantiles))
    else:
        log_ratio_percentiles = {}

    return {
        "loss": safe_metric_mean(losses),
        "approx_kl": safe_metric_mean(approx_kls),
        "clip_frac": safe_metric_mean(clip_fracs),
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "raw_reward_mean": reward_mean,
        "raw_reward_std": reward_std,
        "advantage_mean": float(advantage_values.mean().item()),
        "advantage_std": float(advantage_values.std(unbiased=False).item())
        if advantage_values.numel() > 1
        else 0.0,
        "skipped_updates": skipped_updates,
        "early_stopped": early_stopped,
        "selected_samples": int(batch_size),
        "grad_norm_pre_clip_mean": safe_metric_mean(grad_norms),
        "grad_norm_pre_clip_max": max(grad_norms, default=float("nan")),
        "parameter_update_norm_mean": safe_metric_mean(parameter_update_norms),
        "log_ratio_percentiles": log_ratio_percentiles,
        "timestep_kl": {str(t): safe_metric_mean(values) for t, values in timestep_kls.items()},
        "timestep_clip_frac": {
            str(t): safe_metric_mean(values) for t, values in timestep_clip_fracs.items()
        },
        "reward_by_prompt_category": reward_metrics_by_prompt_category(prompts, reward_values)
        if torch.is_tensor(reward_values)
        else {},
    }


def backward_progressive_interval_length(
    epoch: int,
    total_epochs: int,
    total_steps: int,
    initial_steps: int,
) -> int:
    """Compute the active suffix length for backward progressive training.

    The interval starts on the final denoising steps and expands linearly toward
    the full trajectory as training epochs advance.

    Args:
        epoch: One-indexed training epoch.
        total_epochs: Total number of training epochs in the run.
        total_steps: Number of denoising steps in each trajectory.
        initial_steps: Number of final denoising steps to train on at epoch one.

    Returns:
        The number of final denoising steps to optimize this epoch.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    initial_steps = max(1, min(int(initial_steps), int(total_steps)))
    total_epochs = max(1, int(total_epochs))
    epoch = max(1, min(int(epoch), total_epochs))
    if total_epochs == 1:
        return total_steps
    progress = (epoch - 1) / (total_epochs - 1)
    interval = round(initial_steps + progress * (total_steps - initial_steps))
    return max(1, min(int(interval), int(total_steps)))


def suffix_step_indices(total_steps: int, interval_steps: int) -> list[int]:
    """Return trajectory indices for the final ``interval_steps`` steps.

    Args:
        total_steps: Total denoising steps in a trajectory.
        interval_steps: Number of final steps in the active training interval.

    Returns:
        A list of zero-based indices that select the active suffix.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    interval_steps = max(1, min(int(interval_steps), int(total_steps)))
    return list(range(total_steps - interval_steps, total_steps))


def slice_rollout_steps(rollout: dict[str, Any], step_indices: list[int]) -> dict[str, Any]:
    """Slice rollout tensors to a selected set of denoising-step indices.

    Args:
        rollout: Rollout dictionary containing trajectory tensors with the step
            dimension at index 1, such as ``states`` and ``actions``.
        step_indices: Zero-based step indices to keep.

    Returns:
        A shallow copy of ``rollout`` with step-indexed tensors sliced.
    """
    sliced = dict(rollout)
    for key in ("states", "actions", "old_log_probs", "rewards"):
        value = rollout.get(key)
        if torch.is_tensor(value) and value.ndim >= 2:
            sliced[key] = value[:, step_indices]
    timesteps = rollout.get("timesteps")
    if torch.is_tensor(timesteps):
        sliced["timesteps"] = timesteps[step_indices]
    return sliced


def select_branch_extremes(
    rewards: Tensor,
    branch_size: int,
    min_reward_gap: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select best/worst continuations from each branch group.

    Branch-based sampling creates ``branch_size`` continuations from a shared
    branch state. This helper keeps the highest- and lowest-reward continuation
    from each group and assigns centered, signed advantages to the selected
    samples.

    Args:
        rewards: One-dimensional tensor of terminal rewards ordered by branch
            group, then continuation.
        branch_size: Number of continuations generated per branch group.
        min_reward_gap: Minimum best-minus-worst reward difference required for
            a group to be selected.

    Returns:
        A tuple ``(indices, advantages, reward_pairs)`` where ``indices`` are
        selected flat sample indices, ``advantages`` contains one positive and
        one negative advantage per selected group, and ``reward_pairs`` stores
        each group's ``[best_reward, worst_reward]`` values.
    """
    rewards = rewards.detach().float().flatten()
    if branch_size < 2:
        raise ValueError("branch_size must be at least 2 for branch-based sampling.")
    if rewards.numel() % branch_size != 0:
        raise ValueError("rewards length must be divisible by branch_size.")

    grouped = rewards.reshape(-1, branch_size)
    selected_indices: list[int] = []
    selected_advantages: list[Tensor] = []
    selected_pairs: list[Tensor] = []
    for group_idx, group_rewards in enumerate(grouped):
        best_pos = int(torch.argmax(group_rewards).item())
        worst_pos = int(torch.argmin(group_rewards).item())
        reward_gap = group_rewards[best_pos] - group_rewards[worst_pos]
        if best_pos == worst_pos or float(reward_gap.item()) < min_reward_gap:
            continue
        base_idx = group_idx * branch_size
        half_gap = (reward_gap / 2).clamp_min(1e-8)
        selected_indices.extend([base_idx + best_pos, base_idx + worst_pos])
        selected_advantages.extend([half_gap, -half_gap])
        selected_pairs.append(torch.stack([group_rewards[best_pos], group_rewards[worst_pos]]))

    if not selected_indices:
        device = rewards.device
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, 2, dtype=torch.float32, device=device),
        )
    return (
        torch.tensor(selected_indices, dtype=torch.long, device=rewards.device),
        torch.stack(selected_advantages).to(device=rewards.device, dtype=torch.float32),
        torch.stack(selected_pairs).to(device=rewards.device, dtype=torch.float32),
    )


def standard_eval_prompts(config: Any, limit: int = 10) -> list[str]:
    """Get evaluation prompts from config.
    
    Prefers config.eval_prompts if defined, otherwise falls back to
    deduplicating [config.prompt, *config.train_prompts].
    
    Args:
        config: Baseline config with prompt fields
        limit: Maximum number of prompts to return
        
    Returns:
        List of evaluation prompts, up to limit
    """
    eval_rng = random.Random(123)
    # Prefer explicit eval_prompts when available
    if hasattr(config, "eval_prompts") and config.eval_prompts:
        eval_prompts = eval_rng.sample(list(config.eval_prompts), k=limit)
        return eval_prompts
    
    # Fallback to legacy behavior for backward compatibility
    prompts = [config.prompt, *list(config.train_prompts)]
    deduped = list(dict.fromkeys(prompts))
    return deduped[:limit]


def save_json(path: Path | str, payload: Any) -> None:
    """Save JSON-serializable experiment data with stable formatting.

    Args:
        path: Destination JSON path.
        payload: JSON-serializable object to write.

    Returns:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_image_grid_outputs(
    images: list[Image.Image],
    prompts: list[str],
    output_dir: Path | str,
    prefix: str,
    epoch: int | None = None,
) -> list[dict[str, Any]]:
    """Save images plus a manifest compatible across baseline notebooks.

    Args:
        images: Generated images to write.
        prompts: Prompt strings aligned with ``images``.
        output_dir: Directory that receives the image files and manifest.
        prefix: Filename prefix, such as ``eval`` or ``epoch_0001_sample``.
        epoch: Optional epoch number. If provided, appends to a consolidated manifest.

    Returns:
        A manifest list containing prompt and image path metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    for idx, image in enumerate(images):
        image_path = output_dir / f"{prefix}_{idx:02d}.png"
        image.save(image_path)
        entry = {"index": idx, "prompt": prompts[idx] if idx < len(prompts) else "", "image": str(image_path)}
        if epoch is not None:
            entry["epoch"] = epoch
        manifest_entries.append(entry)
    
    # If epoch is provided, append to consolidated manifest
    if epoch is not None:
        manifest_path = output_dir / "eval_manifest.json"
        existing_manifest = []
        if manifest_path.exists():
            import json
            with manifest_path.open("r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
        existing_manifest.extend(manifest_entries)
        save_json(manifest_path, existing_manifest)
    else:
        # Legacy behavior: create separate manifest file
        save_json(output_dir / f"{prefix}_manifest.json", manifest_entries)
    
    return manifest_entries
