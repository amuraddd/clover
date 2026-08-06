"""Shared runtime helpers for the baseline scripts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

import torch
from PIL import Image
from torch import Tensor

from clover.utils.baseline_utils import (
    save_image_grid_outputs,
    save_json,
    standard_eval_prompts,
)

ConfigT = TypeVar("ConfigT")


def normalize_rewards(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize rewards to have mean 0 and variance 1.

    Args:
        rewards: Tensor of rewards with shape (batch_size, trajectory_length) or any shape
        eps: Small epsilon to prevent division by zero

    Returns:
        Normalized rewards with mean 0 and variance 1
    """
    # Flatten to compute statistics across all rewards
    flat_rewards = rewards.flatten()
    
    # Compute mean and std
    mean = flat_rewards.mean()
    std = flat_rewards.std(unbiased=False)
    
    # Normalize: if std is too small, just center (subtract mean)
    if std < eps:
        return rewards - mean
    
    return (rewards - mean) / std


def parse_config(config_type: type[ConfigT], description: str) -> ConfigT:
    """Build a config from the notebook defaults and common CLI overrides."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-epochs", type=int)
    parser.add_argument("--rollouts-per-epoch", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--adam-epsilon", type=float)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--clip-range", type=float)
    parser.add_argument("--target-kl", type=float)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--reward-type", type=str)
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="*",
        help="CUDA device ids. Pass no values to force CPU.",
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Use float32 even when CUDA is available.",
    )
    args = parser.parse_args()
    overrides: dict[str, Any] = {
        name: value
        for name, value in vars(args).items()
        if value is not None and name != "no_mixed_precision"
    }
    if args.no_mixed_precision:
        overrides["mixed_precision"] = False
    return config_type(**overrides)


def prepare_output(config: Any) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", asdict(config))
    return output_dir


def make_reward_fn(device: torch.device, reward_type: str = "aesthetic"):
    """Return the configured reward function.

    Args:
        device: Torch device for reward computation
        reward_type: Type of reward function to use (default: "aesthetic")

    Returns:
        Callable with signature (images, prompts) -> Tensor
    """
    from clover.utils.rewards_utils import get_reward_fn

    reward_fn_impl = get_reward_fn(reward_type)
    return lambda images, prompts: reward_fn_impl(images, prompts, device)


@torch.no_grad()
def generate_eval_images(
    pipe: Any,
    prompts: list[str],
    config: Any,
    device: torch.device,
    seed: int = 123,
) -> list[Image.Image]:
    pipe.unet.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    images = pipe(
        prompts,
        negative_prompt=[config.negative_prompt] * len(prompts),
        height=config.height,
        width=config.width,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        generator=generator,
    ).images
    pipe.unet.train()
    return images


def evaluate(pipe: Any, config: Any, device: torch.device, epoch: int | None = None) -> None:
    prompts = standard_eval_prompts(config)
    images = generate_eval_images(pipe, prompts, config, device)
    eval_dir = Path(config.output_dir) / "evals"
    image_dir = eval_dir / "images"
    eval_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    image_prefix = f"epoch_{epoch:04d}_eval" if epoch is not None else "eval"
    manifest_entries = save_image_grid_outputs(images, prompts, image_dir, image_prefix, epoch=epoch)

    from clover.utils.rewards_utils import bert_reward, clip_reward

    clip_scores = clip_reward(images, prompts, device=device).tolist()
    bert_scores = bert_reward(images, prompts, device=device).tolist()

    evaluate_metrics = {
        "epoch": epoch,
        "prompts": prompts,
        "image_paths": [entry["image"] for entry in manifest_entries],
        "clip_reward": clip_scores,
        "bert_reward": bert_scores,
    }

    eval_metrics_file = eval_dir / "eval_metrics.json"
    eval_metrics_history: list[dict[str, Any]] = []
    if eval_metrics_file.exists():
        with eval_metrics_file.open("r", encoding="utf-8") as file:
            loaded_metrics = json.load(file)
            if isinstance(loaded_metrics, list):
                eval_metrics_history = loaded_metrics

    eval_metrics_history.append(evaluate_metrics)
    save_json(eval_metrics_file, eval_metrics_history)


def save_trajectory_data(
    baseline_name: str,
    epoch: int,
    rollout: dict[str, Any],
    data_dir: Path | str = "clover/data",
    keep_latest_only: bool = False,
) -> None:
    """Persist an offline-training trajectory to a consolidated PyTorch file.

    By default, trajectories accumulate under monotonically increasing rollout
    numbers. With ``keep_latest_only=True``, the file is atomically replaced
    with a single entry for the current epoch. Tensor values are moved to CPU,
    and images are stored as RGB uint8 tensors.

    Args:
        baseline_name: Name of the baseline (e.g., 'ddpo', 'dpok', 'b2diffurl')
        epoch: Current training epoch number
        rollout: Dictionary containing trajectory data
        data_dir: Base data directory (default: 'clover/data')
        keep_latest_only: Replace prior trajectories instead of appending.
    """
    data_dir = Path(data_dir)
    baseline_data_dir = data_dir / baseline_name
    baseline_data_dir.mkdir(parents=True, exist_ok=True)

    required_fields = ("states", "actions", "prompts", "rewards", "timesteps", "old_log_probs", "images")
    missing_fields = [field for field in required_fields if field not in rollout]
    if missing_fields:
        raise KeyError(f"Cannot save incomplete trajectory; missing fields: {', '.join(missing_fields)}")

    def cpu_copy(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        return value

    def images_to_tensor(images: Any) -> Tensor:
        if torch.is_tensor(images):
            return images.detach().cpu().clone()

        image_tensors = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise TypeError(f"Expected PIL images or an image tensor, got {type(image).__name__}")
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size
            image_tensor = torch.frombuffer(bytearray(rgb_image.tobytes()), dtype=torch.uint8)
            image_tensors.append(image_tensor.reshape(height, width, 3).permute(2, 0, 1).clone())

        if not image_tensors:
            return torch.empty((0, 3, 0, 0), dtype=torch.uint8)
        return torch.stack(image_tensors)

    trajectory_data = {
        "epoch": int(epoch),
        "state": cpu_copy(rollout["states"]),
        "action": cpu_copy(rollout["actions"]),
        "prompts": list(rollout["prompts"]),
        "rewards": cpu_copy(rollout["rewards"]),
        "timesteps": cpu_copy(rollout["timesteps"]),
        "old_log_probs": cpu_copy(rollout["old_log_probs"]),
        "images": images_to_tensor(rollout["images"]),
    }

    trajectories_file = baseline_data_dir / "trajectories.pt"
    existing_trajectories: dict[int, dict[str, Any]] = {}
    if trajectories_file.exists() and not keep_latest_only:
        loaded = torch.load(trajectories_file, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected a trajectory dictionary in {trajectories_file}")
        existing_trajectories.update(loaded)

    rollout_number = (
        int(epoch)
        if keep_latest_only
        else max((int(key) for key in existing_trajectories), default=0) + 1
    )
    existing_trajectories[rollout_number] = trajectory_data
    temporary_file = trajectories_file.with_suffix(".pt.tmp")
    torch.save(existing_trajectories, temporary_file)
    temporary_file.replace(trajectories_file)


def save_evaluation_metrics(
    baseline_name: str,
    epoch: int,
    metrics: dict[str, float],
    images: list[Image.Image] | None = None,
    prompts: list[str] | None = None,
    output_dir: Path | str = None,
) -> None:
    """Append evaluation metrics to a single structured JSON file.

    Args:
        baseline_name: Name of the baseline (e.g., 'ddpo', 'dpok', 'b2diffurl')
        epoch: Current training epoch number
        metrics: Dictionary of evaluation metrics
        images: Optional list of generated evaluation images
        prompts: Optional list of prompts used for evaluation
        output_dir: Base output directory (default: 'outputs/{baseline_name}')
    """
    if output_dir is None:
        output_dir = Path("outputs") / baseline_name
    else:
        output_dir = Path(output_dir)

    eval_dir = output_dir / "training_evals"
    image_dir = eval_dir / "images"
    eval_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Prepare evaluation record
    eval_data = {
        "epoch": epoch,
        "metrics": metrics,
    }

    if prompts:
        eval_data["prompts"] = prompts

    # Keep all epoch metrics in one file for straightforward comparison.
    eval_file = eval_dir / "training_metrics.json"
    evaluations = []
    if eval_file.exists():
        with eval_file.open("r", encoding="utf-8") as file:
            evaluations = json.load(file)

    evaluations.append(eval_data)
    save_json(eval_file, evaluations)

    # Save images if provided
    if images and prompts:
        save_image_grid_outputs(images, prompts, image_dir, f"epoch_{epoch:04d}", epoch=epoch)


def save_training_data(
    baseline_name: str,
    history: list[dict[str, float]],
    data_dir: Path | str = "clover/data",
) -> None:
    """Save complete training history data.

    Args:
        baseline_name: Name of the baseline (e.g., 'ddpo', 'dpok', 'b2diffurl')
        history: List of training metrics per epoch
        data_dir: Base data directory (default: 'clover/data')
    """
    data_dir = Path(data_dir)
    baseline_data_dir = data_dir / baseline_name
    baseline_data_dir.mkdir(parents=True, exist_ok=True)

    # Save complete training history
    history_file = baseline_data_dir / "training_history.json"
    save_json(history_file, history)
