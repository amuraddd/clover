"""Shared runtime helpers for the baseline scripts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

import torch
from PIL import Image

from clover.utils.baseline_utils import (
    save_image_grid_outputs,
    save_json,
    standard_eval_prompts,
)

ConfigT = TypeVar("ConfigT")


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
