"""EMOV2: max diversity denoising diffusion policy optimization.

Converted from ``clover/exp/ddpo.ipynb``. Run with
``python -m clover.baselines.emo_v2``.
"""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import math
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torch import Tensor
from torch.nn import functional as F
from tqdm.auto import trange

from clover.baselines.common import (
    make_reward_fn,
    parse_config,
    prepare_output,
    save_evaluation_metrics,
    save_trajectory_data,
    save_training_data,
)
from clover.utils.baseline_utils import (
    ddpm_mean_std,
    ddpm_transition_variance,
    previous_timestep,
    transition_log_prob,
    unwrap_unet,
    is_stochastic_ddpm_transition,
    decode_latents,
    encode_prompts,
    load_training_checkpoint,
    load_lora_pipeline,
    resolve_gpu_ids,
    sample_prompt_batch,
    save_image_grid_outputs,
    save_json,
    standard_eval_prompts,
    save_lora_weights,
    save_training_checkpoint,
    set_seed,
    trainable_parameters,
    unet_config,
)
from clover.utils.diversity_score import rollout_fid_scores
from clover.utils.prompts import get_prompts

B2_FULL_TRAIN_PROMPTS, B2_FULL_EVAL_PROMPTS = get_prompts(seed=123, save=True)


@dataclass
class EMOV2V2Config:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    output_dir: str = "outputs/emo_v2"
    seed: int = 17
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    use_data_parallel: bool = True
    negative_prompt: str = "blurry, low quality, distorted"
    train_prompts: tuple[str, ...] = B2_FULL_TRAIN_PROMPTS
    eval_prompts: tuple[str, ...] = B2_FULL_EVAL_PROMPTS
    reward_type: str = "clip"
    height: int = 512
    width: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    eta: float = 1.0
    min_log_prob_std: float = 1e-4
    likelihood_scale: float = 1.0
    rollout_chunk_size: int = 64
    rollouts_per_epoch: int = 256
    train_epochs: int = 10
    sac_epochs: int = 2
    gamma: float = field(init=False)
    # Initial reward weight relative to the unit-weight entropy objective. The
    # training loop doubles this value every ten completed epochs.
    reward_scale: float = 10.0
    importance_ratio_clip: float = 1.0
    minibatch_size: int = 32
    learning_rate: float = 3e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("to_v", "to_k", "to_q", "to_out.0")
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    log_every: int = 1
    save_every: int = 5
    evaluate_every: int = 2

    def __post_init__(self) -> None:
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        self.gamma = 1.0 - (1.0 / self.num_inference_steps)


def normalize_rewards_per_timestep(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize each denoising timestep across rollout samples.

    Discounted diffusion returns have a different natural scale at every
    timestep. Normalizing the whole [batch, timestep] tensor together would
    therefore compare early discounted returns with later, less-discounted
    returns. That creates a timestep-dependent bias: early actions tend to get
    negative advantages and late actions positive advantages regardless of the
    relative quality of their generated images. Computing statistics over the
    batch dimension independently for every timestep preserves the ranking of
    samples at that timestep and removes this artificial temporal signal.
    """
    if rewards.ndim != 2:
        raise ValueError(
            "EMO-v2 rewards must have shape [batch, timestep], "
            f"got {tuple(rewards.shape)}"
        )
    centered = rewards - rewards.mean(dim=0, keepdim=True)
    scale = rewards.std(dim=0, unbiased=False, keepdim=True)
    return centered / (scale + eps)


def reward_scale_for_epoch(initial_scale: float, epoch: int) -> float:
    """Double the reward-vs-entropy weight after every ten epochs."""
    if initial_scale <= 0:
        raise ValueError("initial reward scale must be positive")
    if epoch < 1:
        raise ValueError("epoch must be at least 1")
    return float(initial_scale * (2 ** ((epoch - 1) // 10)))


def learning_rate_for_epoch(initial_learning_rate: float, epoch: int) -> float:
    """Reduce the learning rate by ten after every ten epochs."""
    if initial_learning_rate <= 0:
        raise ValueError("initial learning rate must be positive")
    if epoch < 1:
        raise ValueError("epoch must be at least 1")
    return float(initial_learning_rate * (0.1 ** ((epoch - 1) // 10)))


def capped_log_probability_ratio(
    new_log_prob: Tensor,
    old_log_prob: Tensor,
    max_ratio: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Return new_log_prob / old_log_prob with an upper cap.

    A nearly zero old log probability makes the requested quotient undefined,
    so those entries receive the neutral importance weight one.
    """
    if max_ratio <= 0:
        raise ValueError("max_ratio must be positive")
    if new_log_prob.shape != old_log_prob.shape:
        raise ValueError("new and old log probabilities must have matching shapes")
    safe_old = torch.where(
        old_log_prob.abs() < eps,
        torch.ones_like(old_log_prob),
        old_log_prob,
    )
    ratio = new_log_prob.detach() / safe_old
    ratio = torch.where(old_log_prob.abs() < eps, torch.ones_like(ratio), ratio)
    return ratio.clamp(max=max_ratio)

class EMOV2OutputHead(torch.nn.Module):
    """Keep SD 1.5 noise prediction frozen and add a trainable variance head."""

    def __init__(self, noise_head: torch.nn.Conv2d) -> None:
        super().__init__()
        self.noise_head = noise_head
        self.noise_head.requires_grad_(False)
        self.variance_head = torch.nn.Conv2d(
            noise_head.in_channels, noise_head.out_channels,
            noise_head.kernel_size, noise_head.stride, noise_head.padding,
            noise_head.dilation, noise_head.groups, bias=True,
            padding_mode=noise_head.padding_mode,
            device=noise_head.weight.device, dtype=noise_head.weight.dtype,
        )
        torch.nn.init.zeros_(self.variance_head.weight)
        initial_range_value = torch.tensor(-0.9, dtype=torch.float32)
        initial_logit = 2.0 * torch.atanh(initial_range_value)
        torch.nn.init.constant_(self.variance_head.bias, float(initial_logit))
        self.variance_head.float()

    def forward(self, hidden_states: Tensor) -> Tensor:
        noise = self.noise_head(hidden_states)
        variance_logits = self.variance_head(
            hidden_states.to(dtype=self.variance_head.weight.dtype)
        )
        positive_scale = F.softplus(variance_logits.float())
        interpolation_fraction = -torch.expm1(-positive_scale)
        variance_range = 2.0 * interpolation_fraction - 1.0
        return torch.cat((noise, variance_range.to(dtype=noise.dtype)), dim=1)


def _install_learned_variance_head(pipe: Any) -> None:
    """Install EMO-v2 learned variance without changing shared pipeline loading."""
    unet = unwrap_unet(pipe.unet)
    if isinstance(unet.conv_out, EMOV2OutputHead):
        return
    if not isinstance(unet.conv_out, torch.nn.Conv2d):
        raise TypeError("EMO-v2 expects UNet.conv_out to be a Conv2d layer")
    unet.conv_out = EMOV2OutputHead(unet.conv_out)
    unet.register_to_config(out_channels=2 * unet.config.in_channels)


def _variance_head(pipe: Any) -> torch.nn.Conv2d:
    head = unwrap_unet(pipe.unet).conv_out
    if not isinstance(head, EMOV2OutputHead):
        raise TypeError("EMO-v2 learned variance head is not installed")
    return head.variance_head


def _save_variance_head(pipe: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in _variance_head(pipe).state_dict().items()},
        path,
    )


def _load_variance_head(pipe: Any, path: Path) -> None:
    if path.is_file():
        _variance_head(pipe).load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def _predict_emo_v2_chunked(
    pipe: Any, latents: Tensor, timestep: Tensor | int, prompt_embeds: Tensor,
    guidance_scale: float, chunk_size: int,
) -> Tensor:
    """Apply classifier-free guidance to noise, but not learned variance."""
    if chunk_size < 1:
        raise ValueError("rollout_chunk_size must be at least 1")
    negative_embeds, positive_embeds = prompt_embeds.chunk(2, dim=0)
    predictions = []
    for start in range(0, latents.shape[0], chunk_size):
        stop = min(start + chunk_size, latents.shape[0])
        latent_chunk = latents[start:stop]
        model_input = torch.cat((latent_chunk, latent_chunk), dim=0)
        embeddings = torch.cat((negative_embeds[start:stop], positive_embeds[start:stop]), dim=0)
        model_timestep = timestep
        if isinstance(pipe.unet, torch.nn.DataParallel):
            model_timestep = torch.as_tensor(timestep, device=latents.device).expand(model_input.shape[0])
        output = pipe.unet(
            model_input, model_timestep, encoder_hidden_states=embeddings, return_dict=False
        )[0]
        unconditional, conditional = output.chunk(2, dim=0)
        channels = latent_chunk.shape[1]
        guided_noise = unconditional[:, :channels] + guidance_scale * (
            conditional[:, :channels] - unconditional[:, :channels]
        )
        predictions.append(torch.cat((guided_noise, conditional[:, channels:]), dim=1))
    result = torch.cat(predictions, dim=0)
    if not torch.isfinite(result).all():
        raise FloatingPointError("EMO-v2 predicted non-finite noise or variance")
    return result


def _learned_range_variance(
    scheduler: Any, model_output: Tensor, timestep: int, sample: Tensor,
) -> Tensor:
    """Convert a [-1, 1] prediction to learned-range DDPM variance."""
    channels = sample.shape[1]
    if model_output.shape[1] != 2 * channels:
        raise ValueError(
            f"Learned-range variance requires {2 * channels} output channels; "
            f"got {model_output.shape[1]}."
        )
    predicted_range = model_output[:, channels:].float().clamp(-1.0, 1.0)
    t = int(timestep)
    prev_t = previous_timestep(scheduler, t)
    alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=torch.float32)
    alpha_prod_t = alphas_cumprod[t]
    if prev_t >= 0:
        alpha_prod_t_prev = alphas_cumprod[prev_t]
    elif hasattr(scheduler, "final_alpha_cumprod"):
        alpha_prod_t_prev = scheduler.final_alpha_cumprod.to(
            device=sample.device, dtype=torch.float32
        )
    elif hasattr(scheduler, "one"):
        alpha_prod_t_prev = scheduler.one.to(
            device=sample.device, dtype=torch.float32
        )
    else:
        alpha_prod_t_prev = torch.ones(
            (), device=sample.device, dtype=torch.float32
        )
    current_beta = (1 - alpha_prod_t / alpha_prod_t_prev).clamp_min(1e-20)
    posterior_variance = ddpm_transition_variance(
        scheduler, t, device=sample.device
    ).clamp_min(1e-20)
    fraction = (predicted_range + 1.0) / 2.0
    log_variance = (
        fraction * torch.log(current_beta)
        + (1.0 - fraction) * torch.log(posterior_variance)
    )
    return torch.exp(log_variance)


def _emo_v2_step_with_log_prob(
    scheduler: Any, model_output: Tensor, timestep: int, sample: Tensor,
    generator: torch.Generator | None = None, prev_sample: Tensor | None = None,
    eta: float = 1.0, likelihood_scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    mean, _ = ddpm_mean_std(scheduler, model_output, timestep, sample, eta=eta)
    if int(timestep) <= 0:
        if prev_sample is not None:
            raise ValueError("Cannot evaluate log probability for a deterministic transition")
        return mean.to(sample.dtype), torch.zeros(
            sample.shape[0], device=sample.device, dtype=torch.float32
        )
    std = eta * _learned_range_variance(
        scheduler, model_output, timestep, sample
    ).sqrt()
    deterministic = bool((std <= 0).all().item())
    if deterministic:
        std = torch.zeros_like(std)
    if prev_sample is None:
        if deterministic:
            return mean.to(sample.dtype), torch.zeros(sample.shape[0], device=sample.device)
        noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=torch.float32)
        prev_sample = (mean + std * noise).to(sample.dtype)
    elif deterministic:
        raise ValueError("Cannot evaluate log probability for a deterministic transition")
    return prev_sample, transition_log_prob(mean, std, prev_sample, likelihood_scale)


@torch.no_grad()
def collect_rollouts(
    pipe: Any,
    batch_size: int,
    config: EMOV2V2Config,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    reward_fn: Any,
    vae_scale_factor: int,
) -> dict[str, Tensor | list[str] | list[Image.Image]]:
    """Collect one on-policy batch of complete diffusion trajectories.

    The function samples training prompts and initial latents, records every
    denoising transition, decodes the terminal images, evaluates them with the
    configured reward function, and normalizes the resulting rewards.

    Args:
        pipe: Diffusion pipeline containing the UNet, scheduler, and VAE.
        batch_size: Number of trajectories to sample.
        config: EMOV2V2 sampling and training configuration.
        device: Device on which diffusion sampling is performed.
        dtype: Floating-point dtype used by the diffusion model.
        generator: Seeded PyTorch generator used for stochastic sampling.
        reward_fn: Callable accepting images and prompts and returning one
            reward per sample.
        vae_scale_factor: Spatial downsampling factor used to size the latents.

    Returns:
        A live-training rollout dictionary containing prompts, states, actions,
        old log probabilities, scheduler timesteps, raw trajectory rewards, and
        decoded terminal images.
    """
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
        noise_pred = _predict_emo_v2_chunked(
            pipe,
            latents,
            timestep_tensor,
            prompt_embeds,
            config.guidance_scale,
            config.rollout_chunk_size,
        )
        next_latents, log_prob = _emo_v2_step_with_log_prob(
            pipe.scheduler, noise_pred, timestep, latents, generator, eta=config.eta,
            likelihood_scale=config.likelihood_scale,
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
    rewards = torch.zeros(
        (terminal_rewards.shape[0], len(timesteps)), dtype=terminal_rewards.dtype
    )
    rewards[:, -1] = terminal_rewards
    for t in reversed(range(rewards.shape[1] - 1)):
        rewards[:, t] += config.gamma * rewards[:, t + 1]

    return {
        "prompts": prompts,
        "states": torch.stack(states, dim=1),
        "actions": torch.stack(actions, dim=1),
        "old_log_probs": torch.stack(log_probs, dim=1),
        "timesteps": torch.tensor(timesteps, dtype=torch.long),
        "rewards": rewards,
        "images": images,
    }

@torch.no_grad()
def _generate_emo_v2_eval_images(
    pipe: Any, prompts: list[str], config: EMOV2V2Config,
    device: torch.device, dtype: torch.dtype, seed: int = 123,
) -> list[Image.Image]:
    """Evaluate with EMO-v2 guidance applied only to predicted noise."""
    was_training = pipe.unet.training
    pipe.unet.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    prompt_embeds = encode_prompts(
        pipe, prompts, config.negative_prompt, device, dtype
    )
    pipe.scheduler.set_timesteps(config.num_inference_steps, device=device)
    vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    latents = torch.randn(
        (
            len(prompts), unet_config(pipe).in_channels,
            config.height // vae_scale_factor, config.width // vae_scale_factor,
        ),
        generator=generator, device=device, dtype=dtype,
    ) * pipe.scheduler.init_noise_sigma
    try:
        for timestep_tensor in pipe.scheduler.timesteps:
            model_output = _predict_emo_v2_chunked(
                pipe, latents, timestep_tensor, prompt_embeds,
                config.guidance_scale, config.rollout_chunk_size,
            )
            latents, _ = _emo_v2_step_with_log_prob(
                pipe.scheduler, model_output, int(timestep_tensor.item()), latents,
                generator=generator, eta=config.eta,
                likelihood_scale=config.likelihood_scale,
            )
        return decode_latents(pipe, latents)
    finally:
        pipe.unet.train(was_training)


def _evaluate_emo_v2(
    pipe: Any, config: EMOV2V2Config, device: torch.device,
    dtype: torch.dtype, epoch: int | None = None,
) -> None:
    """Generate and persist standard evaluation artifacts with EMO-v2 sampling."""
    prompts = standard_eval_prompts(config)
    images = _generate_emo_v2_eval_images(pipe, prompts, config, device, dtype)
    eval_dir = Path(config.output_dir) / "evals"
    image_dir = eval_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_prefix = f"epoch_{epoch:04d}_eval" if epoch is not None else "eval"
    manifest = save_image_grid_outputs(
        images, prompts, image_dir, image_prefix, epoch=epoch
    )
    from clover.utils.rewards_utils import bert_reward, clip_reward

    clip_scores = clip_reward(images, prompts, device=device).tolist()
    bert_scores = bert_reward(images, prompts, device=device).tolist()

    def summary(scores: list[float]) -> dict[str, float | int]:
        values = np.asarray(scores, dtype=np.float64)
        count = int(values.size)
        mean = float(values.mean()) if count else float("nan")
        std = float(values.std(ddof=1)) if count > 1 else 0.0
        margin = 1.96 * std / math.sqrt(count) if count else float("nan")
        return {
            "mean": mean, "std": std, "count": count,
            "ci95_low": mean - margin, "ci95_high": mean + margin,
        }

    metrics = {
        "epoch": epoch, "prompts": prompts,
        "image_paths": [entry["image"] for entry in manifest],
        "clip_reward": clip_scores, "bert_reward": bert_scores,
        "clip_reward_summary": summary(clip_scores),
        "bert_reward_summary": summary(bert_scores),
    }
    metrics_path = eval_dir / "eval_metrics.json"
    history = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
            if isinstance(loaded, list):
                history = loaded
    history.append(metrics)
    save_json(metrics_path, history)


def emo_v2_combined_rollouts(
    reference_rollout, trajectories=None, diversity_threshold=0.35,
    trajectory_path=None,
    required_trajectory_epoch=None,
):
    """Combine a reference rollout with samples from the latest saved rollout.

    Saved samples are first selected by exact prompt matches with the current
    rollout. Each selected saved image is then compared with every current image
    having the same prompt using singleton Inception distance. A replay sample
    is retained only when its distance exceeds the diversity threshold and its
    terminal reward is greater than or equal to the mean terminal reward of the
    current-batch samples with that exact prompt.

    Args:
        reference_rollout: Current iteration's pre-replay rollout in live or
            saved trajectory field format.
        trajectories: Optional mapping from integer-like rollout numbers to
            saved rollout dictionaries. When omitted, trajectories are loaded
            from ``trajectory_path`` or the default EMOV2 trajectory file.
        trajectory_path: Optional seed-specific replay file used when
            ``trajectories`` is omitted.
        required_trajectory_epoch: Exact saved epoch to combine. When set, a
            missing file or mismatched saved epoch raises an error instead of
            silently training on only the current rollout.
        diversity_threshold: Minimum normalized singleton-FID for a saved
            sample/current sample pair to qualify for replay.

    Returns:
        A live-training rollout containing every reference sample followed by
        qualifying samples from only the latest saved rollout. If no saved
        trajectory exists or no sample qualifies, returns the reference data.

    Raises:
        TypeError: If rollout containers or fields have unsupported types.
        KeyError: If a required rollout field is missing.
        ValueError: If batch shapes, prompt counts, timestep schedules, or
            trajectory keys are invalid.
    """
    if trajectories is None:
        try:
            trajectory_path = trajectory_path or "clover/data/emo_v2/trajectories.pt"
            trajectories = torch.load(trajectory_path, map_location="cpu", weights_only=True)
        except FileNotFoundError:
            if required_trajectory_epoch is not None:
                raise FileNotFoundError(
                    f"Required replay trajectory for epoch {required_trajectory_epoch} "
                    f"was not found at {trajectory_path}"
                )
            return reference_rollout

    def canonicalize(data):
        """Map live and persisted field aliases to one internal representation.

        Args:
            data: Rollout dictionary using either singular saved tensor names
                or plural live-training tensor names.

        Returns:
            A dictionary with canonical singular state and action keys.

        Raises:
            TypeError: If ``data`` is not a dictionary.
            KeyError: If any required rollout field is absent.
        """
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
        """Validate the state batch and its one-to-one prompt alignment.

        Args:
            data: Canonical rollout dictionary.

        Returns:
            The validated batched state tensor.

        Raises:
            ValueError: If states are not batched tensors or prompt count does
                not equal the state batch size.
        """
        states = data["state"]
        if not torch.is_tensor(states) or states.ndim < 2:
            raise ValueError(f"Expected batched states, got {getattr(states, 'shape', type(states))}")
        if not isinstance(data["prompts"], (list, tuple)) or len(data["prompts"]) != states.shape[0]:
            raise ValueError(f"Expected one prompt per state sample ({states.shape[0]})")
        return states

    def select_batch(data, mask):
        """Select masked samples consistently across all rollout fields.

        The shared timestep vector is retained unchanged; tensors and sequence
        fields are indexed along their leading batch dimension.

        Args:
            data: Canonical rollout dictionary to filter.
            mask: One-dimensional Boolean tensor with one value per sample.

        Returns:
            A rollout dictionary containing only selected samples.

        Raises:
            ValueError: If a tensor field does not match the mask batch size.
            TypeError: If a field has an unsupported value type.
        """
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
        """Convert canonical rollout fields to the shared PPO input schema.

        Args:
            data: Canonical rollout dictionary with singular state and action
                keys.

        Returns:
            A dictionary using plural ``states`` and ``actions`` keys expected
            by the shared PPO update.
        """
        return {
            "prompts": data["prompts"], "states": data["state"], "actions": data["action"],
            "old_log_probs": data["old_log_probs"], "timesteps": data["timesteps"],
            "rewards": data["rewards"], "images": data["images"],
        }

    if not isinstance(reference_rollout, dict):
        raise TypeError("reference_rollout must be a dict")
    reference = canonicalize(reference_rollout)
    validate(reference)

    reference_indices_by_prompt = {}
    for index, prompt in enumerate(reference["prompts"]):
        reference_indices_by_prompt.setdefault(prompt, []).append(index)

    reference_rewards = reference["rewards"]
    if not torch.is_tensor(reference_rewards) or reference_rewards.ndim < 1:
        raise ValueError("Reference rewards must be a batched tensor")
    if reference_rewards.shape[0] != len(reference["prompts"]):
        raise ValueError("Reference rewards must have one row per prompt")
    reference_terminal_rewards = (
        reference_rewards
        if reference_rewards.ndim == 1
        else reference_rewards[:, -1]
    ).float()
    reference_reward_mean_by_prompt = {
        prompt: reference_terminal_rewards[
            torch.tensor(indices, device=reference_terminal_rewards.device)
        ].mean()
        for prompt, indices in reference_indices_by_prompt.items()
    }

    accepted = [reference]
    if not isinstance(trajectories, dict):
        raise TypeError("trajectories must be a dict keyed by rollout number")

    valid_trajectory_keys = [
        key for key, saved_rollout in trajectories.items()
        if isinstance(saved_rollout, dict)
    ]
    if not valid_trajectory_keys:
        if required_trajectory_epoch is not None:
            raise ValueError(
                f"Required replay epoch {required_trajectory_epoch} is unavailable; "
                "trajectory file contains no valid rollouts"
            )
        return training_format(reference)

    try:
        trajectories_by_epoch = {int(key): key for key in valid_trajectory_keys}
    except (TypeError, ValueError) as error:
        raise ValueError("trajectory keys must be integer-like rollout numbers") from error

    if required_trajectory_epoch is None:
        latest_trajectory_key = trajectories_by_epoch[max(trajectories_by_epoch)]
    else:
        required_trajectory_epoch = int(required_trajectory_epoch)
        if required_trajectory_epoch not in trajectories_by_epoch:
            raise ValueError(
                f"Required replay epoch {required_trajectory_epoch} is unavailable; "
                f"found epochs {sorted(trajectories_by_epoch)}"
            )
        latest_trajectory_key = trajectories_by_epoch[required_trajectory_epoch]

    saved_rollout = trajectories[latest_trajectory_key]
    if isinstance(saved_rollout, dict):
        rollout = canonicalize(saved_rollout)
        validate(rollout)
        if not torch.equal(reference["timesteps"], rollout["timesteps"]):
            if required_trajectory_epoch is not None:
                raise ValueError(
                    f"Replay epoch {required_trajectory_epoch} has a different timestep schedule"
                )
            return training_format(reference)
        prompt_mask = torch.tensor(
            [prompt in reference_indices_by_prompt for prompt in rollout["prompts"]],
            dtype=torch.bool,
        )
        if prompt_mask.any():
            prompt_matched_rollout = select_batch(rollout, prompt_mask)
            comparison_mask = torch.tensor(
                [
                    [saved_prompt == current_prompt for current_prompt in reference["prompts"]]
                    for saved_prompt in prompt_matched_rollout["prompts"]
                ],
                dtype=torch.bool,
            )
            fid_scores = rollout_fid_scores(
                reference["images"],
                prompt_matched_rollout["images"],
                comparison_mask=comparison_mask,
                batch_size=4,
            )
            replay_rewards = prompt_matched_rollout["rewards"]
            if not torch.is_tensor(replay_rewards) or replay_rewards.ndim < 1:
                raise ValueError("Replay rewards must be a batched tensor")
            replay_terminal_rewards = (
                replay_rewards
                if replay_rewards.ndim == 1
                else replay_rewards[:, -1]
            ).float()
            reward_thresholds = torch.stack(
                [
                    reference_reward_mean_by_prompt[prompt]
                    for prompt in prompt_matched_rollout["prompts"]
                ]
            ).to(
                device=replay_terminal_rewards.device,
                dtype=replay_terminal_rewards.dtype,
            )
            reward_keep = replay_terminal_rewards >= reward_thresholds
            diversity_keep = fid_scores > diversity_threshold
            keep = diversity_keep & reward_keep.to(device=fid_scores.device)
            if keep.any():
                accepted.append(select_batch(prompt_matched_rollout, keep))

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

def _learned_range_ddpm_entropy(scheduler, model_output, sample, timestep):
    """
    Analytic entropy of

        pi_theta(x_{t-1} | x_t)
            = N(mu_theta, diag(variance_theta))

    for DDPMScheduler(variance_type="learned_range").

    Returns:
        entropy: Tensor [batch_size]

    Notes:
        - model_output must have 2*C channels:
              [prediction, predicted_variance]
        - The prediction is interpolated between posterior and current-beta log variance.
        - At t=0 DDPM adds no Gaussian noise, so we return zero entropy
          contribution for that step.
    """
    channels = sample.shape[1]

    if model_output.shape[1] != 2 * channels:
        raise ValueError(
            "variance_type='learned_range' requires model output with 2x "
            f"the sample channels. Got model_output.shape={model_output.shape}, "
            f"sample.shape={sample.shape}."
        )

    # DDPM does not add stochastic variance at the final t=0 step.
    if int(timestep) <= 0:
        return torch.zeros(
            sample.shape[0],
            device=sample.device,
            dtype=torch.float32,
        )

    # Use the exact variance used by the learned-range transition.
    variance = _learned_range_variance(scheduler, model_output, timestep, sample)

    if not torch.isfinite(variance).all():
        raise FloatingPointError(
            "non-finite learned-range variance"
        )

    # For variance_type="learned_range", Diffusers eventually takes sqrt(variance),
    # so variance must be strictly positive.
    if (variance <= 0).any():
        raise FloatingPointError(
            "learned-range DDPM variance must be strictly positive"
        )

    # Diagonal Gaussian differential entropy per dimension:
    #
    # H = 1/2 * [1 + log(2*pi) + log(sigma^2)]
    entropy_per_dim = 0.5 * (
        1.0
        + math.log(2.0 * math.pi)
        + torch.log(variance)
    )

    # [B, C, H, W] -> [B, C*H*W]
    entropy_per_dim = entropy_per_dim.flatten(1)

    # Mean entropy per latent dimension.
    #
    # Use .sum(dim=1) instead if your log_prob represents the
    # summed joint log-probability over all latent dimensions.
    entropy = entropy_per_dim.mean(dim=1)

    return entropy


def sac_update(
    pipe, rollout, optimizer, config, device, dtype,
    reward_scale: float | None = None,
):
    beta = config.reward_scale if reward_scale is None else reward_scale
    if beta <= 0:
        raise ValueError("reward_scale must be positive")

    if config.importance_ratio_clip <= 0:
        raise ValueError("importance_ratio_clip must be positive")

    if pipe.scheduler.config.variance_type != "learned_range":
        raise ValueError(
            "This update expects "
            "pipe.scheduler.config.variance_type == 'learned_range'"
        )

    states = rollout["states"]
    actions = rollout["actions"]
    old_log_probs = rollout["old_log_probs"]
    timesteps = rollout["timesteps"].tolist()
    prompts = rollout["prompts"]
    rewards = rollout["rewards"]

    batch_size = old_log_probs.shape[0]

    if batch_size == 0:
        return {
            "loss": float("nan"),
            "policy_loss": float("nan"),
            "entropy": float("nan"),
            "entropy_bonus": float("nan"),
            "importance_ratio": float("nan"),
            "reward_mean": float("nan"),
            "reward_std": float("nan"),
            "soft_q_mean": float("nan"),
            "grad_norm_mean": float("nan"),
            "skipped_updates": 0,
            "selected_samples": 0,
        }

    # Beta scales reward relative to the unit-weight entropy term: beta * R + H.
    # Normalize over samples separately at each denoising timestep. See
    # normalize_rewards_per_timestep() for why global trajectory normalization
    # introduces an artificial temporal advantage signal.
    soft_q = beta * normalize_rewards_per_timestep(rewards)

    trajectory_len = old_log_probs.shape[1]
    indices = torch.arange(batch_size)

    losses = []
    policy_losses = []
    entropies = []
    entropy_bonuses = []
    importance_ratios = []
    grad_norms = []

    skipped_updates = 0

    pipe.unet.train()

    for _ in range(config.sac_epochs):
        permutation = indices[
            torch.randperm(batch_size)
        ]

        for start in range(
            0,
            batch_size,
            config.minibatch_size,
        ):
            mb_idx = permutation[
                start:start + config.minibatch_size
            ]

            mb_prompts = [
                prompts[index]
                for index in mb_idx.tolist()
            ]

            prompt_embeds = encode_prompts(
                pipe,
                mb_prompts,
                config.negative_prompt,
                device,
                dtype,
            )

            optimizer.zero_grad(set_to_none=True)

            try:
                for step_idx, timestep in enumerate(timesteps):
                    state = states[
                        mb_idx,
                        step_idx,
                    ].to(
                        device=device,
                        dtype=dtype,
                    )

                    action = actions[
                        mb_idx,
                        step_idx,
                    ].to(
                        device=device,
                        dtype=dtype,
                    )

                    timestep_tensor = torch.tensor(
                        timestep,
                        device=device,
                        dtype=torch.long,
                    )

                    # -----------------------------------------------------
                    # Model output
                    #
                    # With variance_type="learned_range":
                    #
                    # noise_pred.shape == [B, 2*C, H, W]
                    #
                    # first C channels:
                    #     epsilon / x0 / v prediction
                    #
                    # second C channels:
                    #     learned variance
                    # -----------------------------------------------------
                    noise_pred = _predict_emo_v2_chunked(
                        pipe,
                        state,
                        timestep_tensor,
                        prompt_embeds,
                        config.guidance_scale,
                        config.rollout_chunk_size,
                    )

                    expected_channels = 2 * state.shape[1]

                    if noise_pred.shape[1] != expected_channels:
                        raise ValueError(
                            "Learned variance requires a 2C-channel "
                            f"UNet output. Expected {expected_channels}, "
                            f"got {noise_pred.shape[1]}."
                        )

                    # -----------------------------------------------------
                    # Current policy log probability
                    #
                    # IMPORTANT:
                    # EMO-v2 transition helper understands
                    # variance_type='learned_range' and use the SECOND HALF
                    # of noise_pred as the transition variance.
                    #
                    # Pass the full 2C output.
                    # -----------------------------------------------------
                    _, log_prob = _emo_v2_step_with_log_prob(
                        pipe.scheduler,
                        noise_pred,
                        timestep,
                        state,
                        prev_sample=action,
                        eta=config.eta,
                        likelihood_scale=config.likelihood_scale,
                    )

                    if not torch.isfinite(log_prob).all():
                        raise FloatingPointError(
                            "non-finite log probability"
                        )

                    # -----------------------------------------------------
                    # Requested capped log-probability quotient
                    #
                    # rho = log pi_theta(a|s) / log pi_old(a|s)
                    #
                    # Detach rho so it remains a fixed score-function weight
                    # rather than another differentiable surrogate term.
                    # -----------------------------------------------------
                    old = old_log_probs[
                        mb_idx,
                        step_idx,
                    ].to(
                        device=device,
                        dtype=log_prob.dtype,
                    )

                    ratio = capped_log_probability_ratio(
                        log_prob, old,
                        max_ratio=config.importance_ratio_clip,
                    )

                    # -----------------------------------------------------
                    # Q / advantage
                    # -----------------------------------------------------
                    q_value = soft_q[
                        mb_idx,
                        step_idx,
                    ].to(
                        device=device,
                        dtype=log_prob.dtype,
                    ).detach()

                    # -----------------------------------------------------
                    # Reward policy gradient surrogate
                    #
                    # Gradient:
                    #
                    # rho * Q * grad log pi_theta
                    # -----------------------------------------------------
                    policy_objective = (
                        ratio.detach()
                        * q_value
                        * log_prob
                    ).mean()

                    # -----------------------------------------------------
                    # Exact Gaussian entropy from learned variance
                    #
                    # H(pi_theta(. | s_t))
                    #
                    # This remains attached to the variance prediction,
                    # so autograd computes:
                    #
                    # grad_theta H
                    # -----------------------------------------------------
                    entropy = _learned_range_ddpm_entropy(
                        pipe.scheduler,
                        noise_pred,
                        state,
                        timestep,
                    )

                    entropy_objective = entropy.mean()

                    # -----------------------------------------------------
                    # Entropy-regularized objective:
                    #
                    # J =
                    #   rho * Q * log pi
                    #   + H(pi)
                    #
                    # Gradient:
                    #
                    # rho Q grad log pi
                    # + grad H
                    # -----------------------------------------------------
                    objective = (
                        policy_objective
                        + entropy_objective
                    )

                    step_loss = (
                        -objective / trajectory_len
                    )

                    if not torch.isfinite(step_loss):
                        raise FloatingPointError(
                            "non-finite entropy-regularized loss"
                        )

                    step_loss.backward()

                    # -----------------------------------------------------
                    # Metrics
                    # -----------------------------------------------------
                    losses.append(
                        float(
                            objective
                            .detach()
                            .neg()
                            .cpu()
                        )
                    )

                    policy_losses.append(
                        float(
                            policy_objective
                            .detach()
                            .cpu()
                        )
                    )

                    entropies.append(
                        float(
                            entropy_objective
                            .detach()
                            .cpu()
                        )
                    )

                    entropy_bonuses.append(
                        float(
                            (
                                entropy_objective
                            )
                            .detach()
                            .cpu()
                        )
                    )

                    importance_ratios.append(
                        float(
                            ratio
                            .mean()
                            .detach()
                            .cpu()
                        )
                    )

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(pipe.unet),
                    config.max_grad_norm,
                )

                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        "non-finite entropy-regularized gradients"
                    )

                grad_norms.append(
                    float(
                        grad_norm
                        .detach()
                        .cpu()
                    )
                )

                optimizer.step()

            except FloatingPointError as error:
                optimizer.zero_grad(set_to_none=True)

                skipped_updates += 1

                print(
                    f"Skipped minibatch: {error}"
                )

            if device.type == "cuda":
                torch.cuda.empty_cache()

    reward_values = rewards.float()

    metric_mean = lambda values: (
        float(np.mean(values))
        if values
        else float("nan")
    )

    return {
        "loss": metric_mean(losses),
        "policy_loss": metric_mean(policy_losses),
        "entropy": metric_mean(entropies),
        "entropy_bonus": metric_mean(entropy_bonuses),
        "beta": float(beta),
        "reward_scale": float(beta),
        "importance_ratio": metric_mean(
            importance_ratios
        ),
        "reward_mean": float(
            reward_values.mean()
        ),
        "reward_std": float(
            reward_values.std(
                unbiased=False
            )
        ),
        "soft_q_mean": float(
            soft_q.float().mean()
        ),
        "grad_norm_mean": metric_mean(
            grad_norms
        ),
        "skipped_updates": skipped_updates,
        "selected_samples": batch_size,
    }
    
def train(
    config: EMOV2V2Config,
    baseline_name: str = "emo_v2",
) -> list[dict[str, float]]:
    """Train an EMOV2-V2 LoRA policy and persist its experiment artifacts.

    Each iteration collects a reference rollout, augments it with qualifying
    samples from only the preceding saved rollout, performs a SAC update, and
    saves only the uncombined reference rollout for the next iteration.

    Args:
        config: Complete EMOV2 model, sampling, optimization, evaluation, and
            output configuration.

    Returns:
        One metrics dictionary per completed training epoch.
    """
    output_dir = prepare_output(config)
    gpu_ids = resolve_gpu_ids(config)
    device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
    dtype = (
        torch.float16
        if device.type == "cuda" and config.mixed_precision
        else torch.float32
    )
    generator = set_seed(config.seed, device)
    print(device, dtype, f"gpu_ids={gpu_ids}")
    reward_fn = make_reward_fn(device, config.reward_type)
    pipe = load_lora_pipeline(config, device=device, dtype=dtype, gpu_ids=gpu_ids)
    _install_learned_variance_head(pipe)
    pipe.scheduler = type(pipe.scheduler).from_config(
        pipe.scheduler.config, variance_type="learned_range", clip_sample=False
    )
    if (
        pipe.scheduler.config.variance_type != "learned_range"
        or pipe.scheduler.variance_type != "learned_range"
    ):
        raise RuntimeError("Failed to configure EMO-v2 learned_range scheduler")
    parameters = trainable_parameters(pipe.unet)
    print(f"Training {sum(parameter.numel() for parameter in parameters):,} LoRA parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train_epochs, eta_min=1e-6
    )
    vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    last_epoch, history = load_training_checkpoint(
        pipe, optimizer, output_dir, device, generator, scheduler=scheduler
    )
    _load_variance_head(pipe, output_dir / "checkpoint" / "variance_head.pt")
    for epoch in trange(last_epoch + 1, config.train_epochs + 1):
        # Derive the reward scale from the absolute epoch so resumed runs use the
        # same value even when loading a checkpoint created before the schedule
        # was introduced.
        scheduled_reward_scale = reward_scale_for_epoch(
            config.reward_scale, epoch
        )
        generator = set_seed(config.seed + epoch, device)
        reference_rollout = collect_rollouts(
            pipe, config.rollouts_per_epoch, config, device, dtype, generator, reward_fn, vae_scale_factor
        )

        data_name = f"{baseline_name}/seed_{config.seed}"
        combined_rollout = emo_v2_combined_rollouts(
            reference_rollout, diversity_threshold=0.3,
            trajectory_path=f"clover/data/{data_name}/trajectories.pt",
            required_trajectory_epoch=epoch - 1 if epoch > 1 else None,
        )

        # apply SAC update to the model using the collected rollouts and save the metrics to history
        metrics = sac_update(
            pipe, combined_rollout, optimizer, config, device, dtype,
            reward_scale=scheduled_reward_scale,
        )
        reference_count = int(reference_rollout["rewards"].shape[0])
        combined_rewards = combined_rollout["rewards"][:, 0].float()
        current_rewards = combined_rewards[:reference_count]
        replay_rewards = combined_rewards[reference_count:]
        metrics.update(
            current_reward_mean=float(current_rewards.mean()),
            current_reward_std=float(current_rewards.std(unbiased=False)),
            replay_reward_mean=float(replay_rewards.mean())
            if replay_rewards.numel()
            else float("nan"),
            replay_reward_std=float(replay_rewards.std(unbiased=False))
            if replay_rewards.numel() > 1
            else (0.0 if replay_rewards.numel() else float("nan")),
            replay_samples=int(replay_rewards.numel()),
            replay_source_epoch=epoch - 1 if epoch > 1 else None,
        )
        metrics["epoch"] = epoch
        metrics["seed"] = config.seed
        metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.append(metrics)
        save_json(output_dir / "history.json", history)

        # Persist only the current iteration's pre-replay reference rollout.
        save_trajectory_data(data_name, epoch, reference_rollout, keep_latest_only=True)

        # Save evaluation metrics from the current epoch training
        save_evaluation_metrics(
            baseline_name,
            epoch,
            metrics,
            reference_rollout.get("images"),
            reference_rollout.get("prompts"),
            output_dir,
        )

        if epoch % config.log_every == 0:
            print(metrics)
        if epoch % config.save_every == 0:
            save_training_checkpoint(
                pipe, optimizer, output_dir, epoch, history, generator, scheduler=scheduler
            )
            _save_variance_head(pipe, output_dir / "checkpoint" / "variance_head.pt")
        del reference_rollout, combined_rollout
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if config.evaluate_every > 0 and epoch % config.evaluate_every == 0:
            _evaluate_emo_v2(pipe, config, device, dtype, epoch=epoch)

    # Save final training data
    save_training_data(f"{baseline_name}/seed_{config.seed}", history)

    final_dir = output_dir / "lora_final"
    save_lora_weights(pipe, final_dir)
    _save_variance_head(pipe, final_dir / "variance_head.pt")
    save_json(output_dir / "config.json", asdict(config))
    save_json(output_dir / "history.json", history)
    print(f"Saved fine-tuned LoRA weights to {final_dir}")
    return history


def main() -> None:
    """Parse command-line configuration and run EMOV2 training.

    Returns:
        None.
    """
    train(parse_config(EMOV2V2Config, __doc__ or "Train EMOV2V2"))


if __name__ == "__main__":
    main()
