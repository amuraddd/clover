"""MD3PO-SAC: max diversity denoising diffusion policy optimization.

Converted from ``clover/exp/ddpo.ipynb``. Run with
``python -m clover.baselines.md3po_sac``.
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
    parse_config,
    prepare_output,
    save_evaluation_metrics,
    save_trajectory_data,
    save_training_data,
)
from clover.utils.baseline_utils import (
    ddpm_step_with_log_prob,
    is_stochastic_ddpm_transition,
    decode_latents,
    encode_prompts,
    load_training_checkpoint,
    load_lora_pipeline,
    ddpm_step_with_log_prob, encode_prompts, normalize_advantages, predict_noise_chunked,
    predict_noise_chunked,
    resolve_gpu_ids,
    sample_prompt_batch,
    save_json,
    save_lora_weights,
    save_training_checkpoint,
    set_seed,
    trainable_parameters,
    unet_config,
)
from clover.utils.diversity_score import rollout_fid_scores
from clover.utils.rewards_utils import (
    clip_prompt_embeddings,
    load_clip_encoder,
)
from clover.utils.prompts import get_prompts

B2_FULL_TRAIN_PROMPTS, B2_FULL_EVAL_PROMPTS = get_prompts(seed=123, save=True)


@dataclass
class MD3POSACConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    output_dir: str = "outputs/md3po_sac"
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
    reward_scale: float = 20.0
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


@torch.no_grad()
def collect_rollouts(
    pipe: Any,
    batch_size: int,
    config: MD3POSACConfig,
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
        config: MD3PO sampling and training configuration.
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
        noise_pred = predict_noise_chunked(
            pipe,
            latents,
            timestep_tensor,
            prompt_embeds,
            config.guidance_scale,
            config.rollout_chunk_size,
        )
        next_latents, log_prob = ddpm_step_with_log_prob(
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


def calculate_ssim(image_a, image_b):
    """Calculate grayscale structural similarity between two color images.

    Args:
        image_a: First OpenCV-compatible image array.
        image_b: Second OpenCV-compatible image array with the same shape as
            ``image_a``.

    Returns:
        The scalar structural similarity index produced by scikit-image.
    """
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray_a, gray_b, full=True)
    return score


def _to_rgba_uint8(frame_chw):
    """Convert one tensor or array frame to an HWC ``uint8`` RGBA array.

    Channel-first inputs are transposed, grayscale and RGB inputs receive the
    required channel expansion, and floating-point values are min-max scaled
    to the range 0–255.

    Args:
        frame_chw: Tensor or NumPy-compatible array representing one image,
            normally in channel-first layout.

    Returns:
        A NumPy ``uint8`` array in height-width-channel RGBA layout.
    """
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


def md3po_combined_rollouts(
    reference_rollout, trajectories=None, diversity_threshold=0.35,
    prompt_similarity_threshold=0.9, trajectory_path=None,
):
    """Combine a reference rollout with samples from the latest saved rollout.

    Every saved sample from the immediately preceding iteration is compared
    with every current sample using CLIP prompt similarity and normalized
    Inception-feature FID. A saved sample is retained when at least one current
    sample has sufficiently similar prompt semantics and an FID below the
    configured threshold.

    Args:
        reference_rollout: Current iteration's pre-replay rollout in live or
            saved trajectory field format.
        trajectories: Optional mapping from integer-like rollout numbers to
            saved rollout dictionaries. When omitted, trajectories are loaded
            from ``trajectory_path`` or the default MD3PO-SAC trajectory file.
        trajectory_path: Optional seed-specific replay file used when
            ``trajectories`` is omitted.
        diversity_threshold: Minimum normalized singleton-FID for a saved
            sample/current sample pair to qualify for replay.
        prompt_similarity_threshold: Minimum cosine similarity between aligned
            prompt embeddings for a saved sample to be retained.

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
            trajectory_path = trajectory_path or "clover/data/md3po_sac/trajectories.pt"
            trajectories = torch.load(trajectory_path, map_location="cpu", weights_only=True)
        except FileNotFoundError:
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

    clip_model, _, clip_tokenizer, clip_device = load_clip_encoder()

    def encode_prompts(prompts):
        """Encode prompt strings as normalized CLIP text embeddings.

        Args:
            prompts: Sequence of prompt strings to encode.

        Returns:
            A two-dimensional tensor containing one embedding per prompt.
        """
        return clip_prompt_embeddings(
            prompts,
            model=clip_model,
            tokenizer=clip_tokenizer,
            device=clip_device,
            batch_size=4,
        )

    reference_prompt_encodings = encode_prompts(reference["prompts"])

    accepted = [reference]
    if not isinstance(trajectories, dict):
        raise TypeError("trajectories must be a dict keyed by rollout number")

    valid_trajectory_keys = [
        key for key, saved_rollout in trajectories.items()
        if isinstance(saved_rollout, dict)
    ]
    if not valid_trajectory_keys:
        return training_format(reference)

    try:
        latest_trajectory_key = max(valid_trajectory_keys, key=lambda key: int(key))
    except (TypeError, ValueError) as error:
        raise ValueError("trajectory keys must be integer-like rollout numbers") from error

    saved_rollout = trajectories[latest_trajectory_key]
    if isinstance(saved_rollout, dict):
        rollout = canonicalize(saved_rollout)
        validate(rollout)
        if not torch.equal(reference["timesteps"], rollout["timesteps"]):
            return training_format(reference)
        previous_prompt_encodings = encode_prompts(rollout["prompts"])
        prompt_scores = previous_prompt_encodings @ reference_prompt_encodings.T
        prompt_matches = prompt_scores >= prompt_similarity_threshold
        fid_scores = rollout_fid_scores(
            reference["images"],
            rollout["images"],
            comparison_mask=prompt_matches,
            device=clip_device,
            batch_size=4,
        )
        keep = fid_scores > diversity_threshold
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


def sac_update(pipe, rollout, optimizer, config, device, dtype):
    """Apply an off-policy maximum-entropy actor update to the diffusion UNet.

    Each denoising transition is an action sampled from the conditional policy
    ``pi_theta(a_t | s_t, prompt)``. Since the image reward is terminal, its
    normalized return is used as a Monte Carlo Q estimate and multiplied by the
    SAC reward scale ``beta = reward_scale``::

        Q_hat_i = (R_i - mean(R)) / (std(R) + 1e-8)
        Q_scaled_i = beta * Q_hat_i

    This follows the reward-scaling formulation used in SAC: the entropy term
    has fixed unit weight, while ``beta`` controls reward relative to entropy.
    Maximizing ``beta * E[Q] + H(pi)`` is equivalent, up to division by beta,
    to maximizing ``E[Q] + alpha * H(pi)`` with ``alpha = 1 / beta``. Therefore
    a smaller reward scale produces a higher effective temperature and favors
    more entropy; a larger reward scale emphasizes terminal image reward.

    MD3PO filtering can include transitions from an earlier behavior policy
    ``mu``. The update corrects those samples with a bounded importance ratio::

        rho_i,t = min(exp(log pi_theta(a_i,t | s_i,t)
                          - log mu(a_i,t | s_i,t)), rho_max)

    The implementation detaches ``rho_i,t`` and every coefficient multiplying
    the current log probability. Each transition minimizes::

        L_i,t(theta) = -rho_i,t * [Q_scaled_i
                         - (log pi_theta(a_i,t | s_i,t) + 1)]
                         * log pi_theta(a_i,t | s_i,t)

    The unit-weight entropy term follows from the score-function identity::

        grad H(pi) = -E_pi[(log pi + 1) * grad log pi]

    The minibatch loss is divided by the number of denoising transitions before
    backpropagation. Unlike PPO, there is no clipped policy objective or KL
    early stopping; clipping applies only to the off-policy importance weight.

    Args:
        pipe: Diffusion pipeline whose trainable LoRA UNet is the actor.
        rollout: Current and filtered replay trajectories containing states,
            actions, rewards, behavior log probabilities, timesteps, and prompts.
        optimizer: Optimizer over trainable diffusion-model parameters.
        config: SAC reward-scaling, sampling, and gradient configuration.
        device: Device used to recompute transition likelihoods.
        dtype: Floating-point dtype used for model inputs.

    Returns:
        Aggregate policy loss, entropy, importance-ratio, reward, gradient, and
        skipped-update metrics for the completed SAC update.
    """
    if config.reward_scale <= 0:
        raise ValueError("reward_scale must be positive")
    if config.importance_ratio_clip <= 0:
        raise ValueError("importance_ratio_clip must be positive")

    states = rollout["states"]
    actions = rollout["actions"]
    old_log_probs = rollout["old_log_probs"]
    timesteps = rollout["timesteps"].tolist()
    prompts = rollout["prompts"]
    rewards = rollout["rewards"]
    batch_size = old_log_probs.shape[0]
    if batch_size == 0:
        return {
            "loss": float("nan"), "policy_loss": float("nan"),
            "entropy": float("nan"), "importance_ratio": float("nan"),
            "reward_mean": float("nan"), "reward_std": float("nan"),
            "soft_q_mean": float("nan"), "skipped_updates": 0,
            "selected_samples": 0,
        }

    soft_q = config.reward_scale * normalize_advantages(rewards)
    trajectory_len = old_log_probs.shape[1]
    indices = torch.arange(batch_size)
    losses, policy_losses, entropies, importance_ratios, grad_norms = [], [], [], [], []
    skipped_updates = 0
    pipe.unet.train()

    for _ in range(config.sac_epochs):
        permutation = indices[torch.randperm(batch_size)]
        for start in range(0, batch_size, config.minibatch_size):
            mb_idx = permutation[start:start + config.minibatch_size]
            mb_prompts = [prompts[index] for index in mb_idx.tolist()]
            prompt_embeds = encode_prompts(pipe, mb_prompts, config.negative_prompt, device, dtype)
            optimizer.zero_grad(set_to_none=True)
            try:
                for step_idx, timestep in enumerate(timesteps):
                    state = states[mb_idx, step_idx].to(device=device, dtype=dtype)
                    action = actions[mb_idx, step_idx].to(device=device, dtype=dtype)
                    timestep_tensor = torch.tensor(timestep, device=device, dtype=torch.long)
                    noise_pred = predict_noise_chunked(
                        pipe, state, timestep_tensor, prompt_embeds,
                        config.guidance_scale, config.rollout_chunk_size,
                    )
                    _, log_prob = ddpm_step_with_log_prob(
                        pipe.scheduler, noise_pred, timestep, state,
                        prev_sample=action, eta=config.eta,
                        likelihood_scale=config.likelihood_scale,
                    )
                    old = old_log_probs[mb_idx, step_idx].to(device=device, dtype=log_prob.dtype)
                    ratio = torch.exp(log_prob.detach() - old).clamp(max=config.importance_ratio_clip)
                    q_value = soft_q[mb_idx, step_idx].to(device=device, dtype=log_prob.dtype)
                    entropy_advantage = -(log_prob.detach() + 1.0)
                    # Reward term: L_Q = -E[rho_t * Q_scaled * log pi_theta(a_t | s_t)].
                    policy_loss = -(ratio * q_value.detach() * log_prob).mean()
                    # Entropy term: L_H = -E[rho_t * (-(log pi_theta + 1)) * log pi_theta].
                    entropy_loss = -(ratio * entropy_advantage * log_prob).mean()
                    # Per-transition objective: L_t = (L_Q + L_H) / T.
                    step_loss = (policy_loss + entropy_loss) / trajectory_len
                    if not torch.isfinite(step_loss):
                        raise FloatingPointError("non-finite SAC loss")
                    step_loss.backward()
                    losses.append(float((step_loss * trajectory_len).detach().cpu()))
                    policy_losses.append(float(policy_loss.detach().cpu()))
                    entropies.append(float((-log_prob).mean().detach().cpu()))
                    importance_ratios.append(float(ratio.mean().detach().cpu()))
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(pipe.unet), config.max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("non-finite SAC gradients")
                grad_norms.append(float(grad_norm.detach().cpu()))
                optimizer.step()
            except FloatingPointError as error:
                optimizer.zero_grad(set_to_none=True)
                skipped_updates += 1
                print(f"Skipped SAC minibatch: {error}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    reward_values = rewards.float()
    metric_mean = lambda values: float(np.mean(values)) if values else float("nan")
    return {
        "loss": metric_mean(losses), "policy_loss": metric_mean(policy_losses),
        "entropy": metric_mean(entropies),
        "reward_scale": float(config.reward_scale),
        "importance_ratio": metric_mean(importance_ratios),
        "reward_mean": float(reward_values.mean()),
        "reward_std": float(reward_values.std(unbiased=False)),
        "soft_q_mean": float(soft_q.float().mean()),
        "grad_norm_mean": metric_mean(grad_norms),
        "skipped_updates": skipped_updates, "selected_samples": batch_size,
    }

def train(config: MD3POSACConfig) -> list[dict[str, float]]:
    """Train an MD3PO LoRA policy and persist its experiment artifacts.

    Each iteration collects a reference rollout, augments it with qualifying
    samples from only the preceding saved rollout, performs a SAC update, and
    saves only the uncombined reference rollout for the next iteration.

    Args:
        config: Complete MD3PO model, sampling, optimization, evaluation, and
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
    parameters = trainable_parameters(pipe.unet)
    print(f"Training {sum(parameter.numel() for parameter in parameters):,} LoRA parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    last_epoch, history = load_training_checkpoint(
        pipe, optimizer, output_dir, device, generator, scheduler=scheduler
    )
    for epoch in trange(last_epoch + 1, config.train_epochs + 1):
        generator = set_seed(config.seed + epoch, device)
        reference_rollout = collect_rollouts(
            pipe, config.rollouts_per_epoch, config, device, dtype, generator, reward_fn, vae_scale_factor
        )

        data_name = f"md3po_sac/seed_{config.seed}"
        combined_rollout = md3po_combined_rollouts(
            reference_rollout, diversity_threshold=0.5,
            trajectory_path=f"clover/data/{data_name}/trajectories.pt",
        )

        # apply SAC update to the model using the collected rollouts and save the metrics to history
        metrics = sac_update(pipe, combined_rollout, optimizer, config, device, dtype)
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
            "md3po_sac",
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
        del reference_rollout, combined_rollout
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if config.evaluate_every > 0 and epoch % config.evaluate_every == 0:
            evaluate(pipe, config, device, epoch=epoch)

    # Save final training data
    save_training_data(f"md3po_sac/seed_{config.seed}", history)

    final_dir = output_dir / "lora_final"
    save_lora_weights(pipe, final_dir)
    save_json(output_dir / "config.json", asdict(config))
    save_json(output_dir / "history.json", history)
    print(f"Saved fine-tuned LoRA weights to {final_dir}")
    return history


def main() -> None:
    """Parse command-line configuration and run MD3PO training.

    Returns:
        None.
    """
    train(parse_config(MD3POSACConfig, __doc__ or "Train MD3PO-SAC"))


if __name__ == "__main__":
    main()
