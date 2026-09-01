"""SQDF offline text-to-image alignment baseline.

Adapted from the official T2I implementation at
https://github.com/Shin-woocheol/SQDF. Run through ``python main.py`` or with
``python -m clover.baselines.sqdf``.

This keeps SQDF's randomized one-step reparameterized policy gradient,
discounted soft-Q reward, and frozen-reference KL penalty. The optional
consistency-model extension is deliberately omitted so the baseline shares
Clover's Stable Diffusion, prompt, reward, checkpoint, and evaluation setup.
"""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, UNet2DConditionModel
from PIL import Image
from torch import Tensor
from tqdm.auto import trange

from clover.baselines.common import (
    evaluate, make_reward_fn, parse_config, prepare_output,
    save_evaluation_metrics, save_training_data,
)
from clover.utils.baseline_utils import (
    encode_prompts, load_lora_pipeline, load_training_checkpoint,
    predict_noise_chunked, resolve_gpu_ids, sample_prompt_batch, save_json,
    save_lora_weights, save_training_checkpoint, set_seed,
    trainable_parameters, unet_config,
)
from clover.utils.prompts import get_prompts

TRAIN_PROMPTS, EVAL_PROMPTS = get_prompts(seed=123, save=False)


@dataclass
class SQDFConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    output_dir: str = "outputs/sqdf"
    seed: int = 17
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    use_data_parallel: bool = True
    negative_prompt: str = "blurry, low quality, distorted"
    train_prompts: tuple[str, ...] = TRAIN_PROMPTS
    eval_prompts: tuple[str, ...] = EVAL_PROMPTS
    reward_type: str = "clip"
    height: int = 512
    width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    eta: float = 1.0
    # Accepted for compatibility with Clover's shared main.py argument set;
    # SQDF uses DDIM mean KL rather than transition log-probability clipping.
    min_log_prob_std: float = 1e-4
    rollout_chunk_size: int = 8
    rollouts_per_epoch: int = 8
    train_epochs: int = 50
    minibatch_size: int = 1
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    lora_rank: int = 4
    lora_alpha: int = 4
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("to_v", "to_k", "to_q", "to_out.0")
    sqdf_gamma: float = 0.9
    sqdf_alpha: float = 0.01
    sqdf_step_min: int = 0
    sqdf_step_max: int = 49
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    log_every: int = 1
    save_every: int = 5
    evaluate_every: int = 2

    def __post_init__(self) -> None:
        if self.reward_type != "clip":
            raise ValueError("SQDF text-to-image alignment requires reward_type='clip'")
        if not 0 < self.sqdf_gamma <= 1:
            raise ValueError("sqdf_gamma must be in (0, 1]")
        if self.sqdf_alpha < 0:
            raise ValueError("sqdf_alpha must be non-negative")
        if self.sqdf_step_min < 0 or self.sqdf_step_max < self.sqdf_step_min:
            raise ValueError("invalid SQDF timestep-index range")


def _cfg_prediction(
    unet: torch.nn.Module,
    latents: Tensor,
    timesteps: Tensor,
    prompt_embeds: Tensor,
    guidance_scale: float,
) -> Tensor:
    model_input = torch.cat((latents, latents), dim=0)
    model_timesteps = torch.cat((timesteps, timesteps), dim=0)
    prediction = unet(
        model_input, model_timesteps, encoder_hidden_states=prompt_embeds,
        return_dict=False,
    )[0]
    unconditional, conditional = prediction.chunk(2)
    return unconditional + guidance_scale * (conditional - unconditional)


def _ddim_terms(
    scheduler: DDIMScheduler,
    sample: Tensor,
    noise: Tensor,
    timesteps: Tensor,
    eta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return DDIM transition mean, standard deviation, and predicted x0."""
    step = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    previous = timesteps - step
    alphas = scheduler.alphas_cumprod.to(sample.device, dtype=sample.dtype)
    alpha_t = alphas[timesteps].view(-1, 1, 1, 1)
    gathered_previous = alphas[previous.clamp_min(0)].view(-1, 1, 1, 1)
    final_alpha = torch.as_tensor(
        scheduler.final_alpha_cumprod, device=sample.device, dtype=sample.dtype
    )
    alpha_previous = torch.where(
        (previous >= 0).view(-1, 1, 1, 1), gathered_previous, final_alpha
    )
    beta_t = 1 - alpha_t
    predicted_x0 = (sample - beta_t.sqrt() * noise) / alpha_t.sqrt()
    variance = (
        (1 - alpha_previous) / (1 - alpha_t) * (1 - alpha_t / alpha_previous)
    ).clamp_min(0)
    std = eta * variance.sqrt()
    direction = (1 - alpha_previous - std.square()).clamp_min(0).sqrt() * noise
    mean = alpha_previous.sqrt() * predicted_x0 + direction
    return mean, std, predicted_x0


def _tensor_to_pil(images: Tensor) -> list[Image.Image]:
    arrays = (
        images.detach().clamp(0, 1).permute(0, 2, 3, 1).float().cpu().numpy() * 255
    ).round().astype("uint8")
    return [Image.fromarray(array) for array in arrays]


class DifferentiableCLIPReward(torch.nn.Module):
    """OpenCLIP cosine alignment reward retaining image-pixel gradients."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        self.model = model.to(device).eval().requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")
        self.register_buffer(
            "mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1)
        )

    def forward(self, images: Tensor, prompts: list[str]) -> Tensor:
        pixels = F.interpolate(images, (224, 224), mode="bicubic", align_corners=False)
        pixels = (pixels - self.mean.to(pixels.dtype)) / self.std.to(pixels.dtype)
        tokens = self.tokenizer(prompts).to(images.device)
        image_features = F.normalize(self.model.encode_image(pixels).float(), dim=-1)
        with torch.no_grad():
            text_features = F.normalize(self.model.encode_text(tokens).float(), dim=-1)
        return (image_features * text_features).sum(dim=-1)


@torch.no_grad()
def _collect_states(
    pipe: Any,
    prompts: list[str],
    config: SQDFConfig,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    embeds = encode_prompts(pipe, prompts, config.negative_prompt, device, dtype)
    pipe.scheduler.set_timesteps(config.num_inference_steps, device=device)
    vae_scale = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    shape = (
        len(prompts), unet_config(pipe).in_channels,
        config.height // vae_scale, config.width // vae_scale,
    )
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    latents *= pipe.scheduler.init_noise_sigma
    states = []
    for timestep in pipe.scheduler.timesteps:
        states.append(latents.cpu())
        noise = predict_noise_chunked(
            pipe, latents, timestep, embeds, config.guidance_scale,
            config.rollout_chunk_size,
        )
        latents = pipe.scheduler.step(
            noise, timestep, latents, eta=config.eta, generator=generator,
            return_dict=False,
        )[0]
    return torch.stack(states, dim=1), embeds


def sqdf_update(
    pipe: Any,
    reference_unet: torch.nn.Module,
    prompts: list[str],
    optimizer: torch.optim.Optimizer,
    config: SQDFConfig,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    differentiable_reward: DifferentiableCLIPReward,
) -> tuple[dict[str, float], dict[str, Any]]:
    states, embeds = _collect_states(pipe, prompts, config, device, dtype, generator)
    count, horizon = states.shape[:2]
    # The terminal DDIM transition is deterministic (zero variance), so it has
    # no finite Gaussian KL and is excluded from SQDF action selection.
    high = min(config.sqdf_step_max, horizon - 2)
    if config.sqdf_step_min > high:
        raise ValueError("sqdf_step_min exceeds the denoising horizon")
    indices = torch.randint(
        config.sqdf_step_min, high + 1, (count,), generator=generator, device=device
    )
    timesteps = pipe.scheduler.timesteps[indices]
    state = states[torch.arange(count), indices.cpu()].to(device)

    train_noise = _cfg_prediction(
        pipe.unet, state, timesteps, embeds, config.guidance_scale
    )
    with torch.no_grad():
        reference_noise = _cfg_prediction(
            reference_unet, state, timesteps, embeds, config.guidance_scale
        )
    mean, std, _ = _ddim_terms(
        pipe.scheduler, state, train_noise, timesteps, config.eta
    )
    reference_mean, _, _ = _ddim_terms(
        pipe.scheduler, state, reference_noise, timesteps, config.eta
    )
    transition = mean + std * torch.randn(
        mean.shape, generator=generator, device=device, dtype=mean.dtype
    )
    kl = (
        (mean - reference_mean).square().flatten(1).mean(1)
        / std.flatten(1).mean(1).square().clamp_min(1e-8)
    )

    next_indices = (indices + 1).clamp_max(horizon - 1)
    next_timesteps = pipe.scheduler.timesteps[next_indices]
    # Reference weights are frozen, but autograd through its input is essential:
    # this is the reparameterized soft-Q gradient used by SQDF.
    next_reference_noise = _cfg_prediction(
        reference_unet, transition, next_timesteps, embeds,
        config.guidance_scale,
    )
    _, _, predicted_x0 = _ddim_terms(
        pipe.scheduler, transition, next_reference_noise, next_timesteps, 0.0
    )
    terminal = (indices == horizon - 1).view(-1, 1, 1, 1)
    predicted_x0 = torch.where(terminal, transition, predicted_x0)
    decoded = pipe.vae.decode(
        predicted_x0.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor,
        return_dict=False,
    )[0]
    image_tensors = (decoded.float() / 2 + 0.5).clamp(0, 1)
    rewards = differentiable_reward(image_tensors, prompts)
    discounts = config.sqdf_gamma ** (horizon - indices - 1)
    loss = -(discounts * rewards - config.sqdf_alpha * kl).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable_parameters(pipe.unet), config.max_grad_norm
    )
    optimizer.step()

    images = _tensor_to_pil(image_tensors)
    # This is the same normalized CLIP cosine alignment score exposed by
    # Clover's ``clip`` reward, retained here to avoid loading a second ViT-H.
    logged_rewards = rewards.detach().float().cpu()
    metrics = {
        "loss": float(loss.detach()),
        "reward_mean": float(logged_rewards.mean()),
        "reward_std": float(logged_rewards.std(unbiased=False)),
        "differentiable_clip_mean": float(rewards.detach().mean()),
        "kl_mean": float(kl.detach().mean()),
        "discount_mean": float(discounts.float().mean()),
        "grad_norm": float(torch.as_tensor(grad_norm)),
    }
    trajectory = {
        "prompts": prompts,
        "timestep_indices": indices.detach().cpu().tolist(),
        "timesteps": timesteps.detach().cpu().tolist(),
        "rewards": logged_rewards.tolist(),
        "kl": kl.detach().float().cpu().tolist(),
        "images": images,
    }
    return metrics, trajectory


def _save_trajectory_json(
    epoch: int, seed: int, trajectories: list[dict[str, Any]]
) -> None:
    trajectory_dir = Path("clover/data/sqdf/trajectories")
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for trajectory in trajectories:
        records.extend(
            {
                "prompt": prompt,
                "timestep_index": trajectory["timestep_indices"][index],
                "timestep": trajectory["timesteps"][index],
                "reward": trajectory["rewards"][index],
                "kl": trajectory["kl"][index],
            }
            for index, prompt in enumerate(trajectory["prompts"])
        )
    save_json(
        trajectory_dir / f"seed_{seed}_epoch_{epoch:04d}.json",
        {"epoch": epoch, "seed": seed, "samples": records},
    )


def train(config: SQDFConfig) -> list[dict[str, float]]:
    output_dir = prepare_output(config)
    gpu_ids = resolve_gpu_ids(config)
    device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
    dtype = (
        torch.float16
        if device.type == "cuda" and config.mixed_precision
        else torch.float32
    )
    generator = set_seed(config.seed, device)
    # Construct through Clover's registry to validate the configured standard
    # reward entry point. Evaluation below also uses that shared implementation.
    make_reward_fn(device, config.reward_type)
    pipe = load_lora_pipeline(config, device, dtype, gpu_ids)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, clip_sample=False)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.model_id, subfolder="unet", torch_dtype=dtype
    )
    reference_unet.requires_grad_(False).eval().to(device)
    if config.use_data_parallel and len(gpu_ids) > 1:
        reference_unet = torch.nn.DataParallel(
            reference_unet, device_ids=gpu_ids, output_device=gpu_ids[0]
        )
        print(f"Reference UNet wrapped with DataParallel on GPUs {gpu_ids}")
    optimizer = torch.optim.AdamW(
        trainable_parameters(pipe.unet), lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon,
    )
    differentiable_reward = DifferentiableCLIPReward(device)
    last_epoch, history = load_training_checkpoint(
        pipe, optimizer, output_dir, device, generator
    )
    for epoch in trange(last_epoch + 1, config.train_epochs + 1):
        generator = set_seed(config.seed + epoch, device)
        epoch_metrics = []
        trajectories = []
        for start in range(0, config.rollouts_per_epoch, config.minibatch_size):
            prompts = sample_prompt_batch(
                config.train_prompts,
                min(config.minibatch_size, config.rollouts_per_epoch - start),
            )
            metrics, trajectory = sqdf_update(
                pipe, reference_unet, prompts, optimizer, config, device, dtype,
                generator, differentiable_reward,
            )
            epoch_metrics.append(metrics)
            trajectories.append(trajectory)
        summary = {
            key: sum(item[key] for item in epoch_metrics) / len(epoch_metrics)
            for key in epoch_metrics[0]
        }
        summary.update(epoch=epoch, seed=config.seed)
        history.append(summary)
        _save_trajectory_json(epoch, config.seed, trajectories)
        latest = trajectories[-1]
        save_evaluation_metrics(
            "sqdf", epoch, summary, latest["images"], latest["prompts"],
            output_dir,
        )
        save_json(output_dir / "history.json", history)
        if epoch % config.log_every == 0:
            print(summary)
        if epoch % config.save_every == 0:
            save_training_checkpoint(
                pipe, optimizer, output_dir, epoch, history, generator
            )
        if config.evaluate_every > 0 and epoch % config.evaluate_every == 0:
            evaluate(pipe, config, device, epoch=epoch)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_training_data(f"sqdf/seed_{config.seed}", history)
    save_lora_weights(pipe, output_dir / "lora_final")
    save_json(output_dir / "config.json", asdict(config))
    return history


def main() -> None:
    train(parse_config(SQDFConfig, __doc__ or "Train SQDF"))


if __name__ == "__main__":
    main()
