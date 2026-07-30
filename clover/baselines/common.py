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


def evaluate(pipe: Any, config: Any, device: torch.device) -> None:
    prompts = standard_eval_prompts(config)
    images = generate_eval_images(pipe, prompts, config, device)
    save_image_grid_outputs(images, prompts, Path(config.output_dir), "eval")


def save_trajectory_data(
    baseline_name: str,
    epoch: int,
    rollout: dict[str, Any],
    data_dir: Path | str = "clover/data",
) -> None:
    """Save RL trajectory data in structured JSON format to a consolidated file.

    Args:
        baseline_name: Name of the baseline (e.g., 'ddpo', 'dpok', 'b2diffurl')
        epoch: Current training epoch number
        rollout: Dictionary containing trajectory data
        data_dir: Base data directory (default: 'clover/data')
    """
    data_dir = Path(data_dir)
    baseline_data_dir = data_dir / baseline_name
    baseline_data_dir.mkdir(parents=True, exist_ok=True)

    # Extract serializable data from rollout
    trajectory_data = {
        "epoch": epoch,
        "prompts": rollout.get("prompts", []),
        "timesteps": rollout.get("timesteps", []).tolist() if hasattr(rollout.get("timesteps", []), "tolist") else rollout.get("timesteps", []),
    }

    # Add rewards if available
    if "rewards" in rollout and hasattr(rollout["rewards"], "tolist"):
        trajectory_data["rewards"] = rollout["rewards"].tolist()

    # Add statistics
    if "rewards" in rollout:
        rewards = rollout["rewards"]
        if hasattr(rewards, "mean"):
            trajectory_data["reward_stats"] = {
                "mean": float(rewards.mean()),
                "std": float(rewards.std()),
                "min": float(rewards.min()),
                "max": float(rewards.max()),
            }

    # Append to consolidated trajectories file
    trajectories_file = baseline_data_dir / "trajectories.json"
    existing_trajectories = []
    if trajectories_file.exists():
        import json
        with trajectories_file.open("r", encoding="utf-8") as f:
            existing_trajectories = json.load(f)
    
    existing_trajectories.append(trajectory_data)
    save_json(trajectories_file, existing_trajectories)


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

    eval_dir = output_dir / "evals/images/"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Prepare evaluation record
    eval_data = {
        "epoch": epoch,
        "metrics": metrics,
    }

    if prompts:
        eval_data["prompts"] = prompts

    # Keep all epoch metrics in one file for straightforward comparison.
    eval_file = eval_dir / "evaluation_metrics.json"
    evaluations = []
    if eval_file.exists():
        with eval_file.open("r", encoding="utf-8") as file:
            evaluations = json.load(file)

    evaluations.append(eval_data)
    save_json(eval_file, evaluations)

    # Save images if provided
    if images and prompts:
        save_image_grid_outputs(images, prompts, eval_dir, f"epoch_{epoch:04d}", epoch=epoch)


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
