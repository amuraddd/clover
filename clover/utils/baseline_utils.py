from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
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


def decode_latents(pipe: StableDiffusionPipeline, latents: Tensor) -> list[Image.Image]:
    if not torch.isfinite(latents).all():
        raise FloatingPointError("Cannot decode non-finite latents; the rollout/update became unstable.")
    latents = latents / pipe.vae.config.scaling_factor
    with torch.no_grad():
        images = pipe.vae.decode(latents.to(dtype=pipe.vae.dtype)).sample
    if not torch.isfinite(images).all():
        raise FloatingPointError("VAE decoded non-finite images; the latents are unstable.")
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    images = (images * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in images]


def previous_timestep(scheduler: DDPMScheduler, timestep: int) -> int:
    if hasattr(scheduler, "previous_timestep"):
        prev = scheduler.previous_timestep(timestep)
        return int(prev.item() if torch.is_tensor(prev) else prev)
    step = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    return int(timestep) - step


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

    variance = (beta_prod_t_prev / beta_prod_t) * current_beta_t
    variance = variance.clamp(min=1e-12)
    std = (eta * variance.sqrt()).clamp_min(1e-6)
    if not torch.isfinite(std).all():
        raise FloatingPointError("Non-finite DDPM transition std; reduce LR/clip range or disable mixed precision.")
    return mean, std


def transition_log_prob(mean: Tensor, std: Tensor, prev_sample: Tensor) -> Tensor:
    prev_sample = prev_sample.float()
    mean = mean.float()
    std = std.float().clamp_min(1e-6)
    log_prob = -0.5 * ((prev_sample - mean) / std).pow(2) - torch.log(std) - 0.5 * math.log(2 * math.pi)
    if not torch.isfinite(log_prob).all():
        raise FloatingPointError("Non-finite transition log-prob; reduce LR/clip range or disable mixed precision.")
    return log_prob.flatten(1).mean(dim=1)


def ddpm_step_with_log_prob(
    scheduler: DDPMScheduler,
    model_output: Tensor,
    timestep: int,
    sample: Tensor,
    generator: torch.Generator | None = None,
    prev_sample: Tensor | None = None,
    eta: float = 1.0,
) -> tuple[Tensor, Tensor]:
    mean, std = ddpm_mean_std(scheduler, model_output, timestep, sample, eta=eta)
    if prev_sample is None:
        noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=torch.float32)
        prev_sample = (mean + std * noise).to(sample.dtype)
    log_prob = transition_log_prob(mean, std, prev_sample)
    return prev_sample, log_prob


def gaussian_kl(mean: Tensor, std: Tensor, ref_mean: Tensor, ref_std: Tensor) -> Tensor:
    mean = mean.float()
    std = std.float().clamp_min(1e-6)
    ref_mean = ref_mean.float()
    ref_std = ref_std.float().clamp_min(1e-6)
    kl = torch.log(ref_std / std) + (std.pow(2) + (mean - ref_mean).pow(2)) / (2 * ref_std.pow(2)) - 0.5
    if not torch.isfinite(kl).all():
        raise FloatingPointError("Non-finite Gaussian KL; reduce LR or disable mixed precision.")
    return kl.flatten(1).mean(dim=1)


def sample_prompt_batch(train_prompts: tuple[str, ...], batch_size: int) -> list[str]:
    return [random.choice(train_prompts) for _ in range(batch_size)]


def normalize_advantages(rewards: Tensor) -> Tensor:
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
    return float(np.mean(values)) if values else float("nan")
