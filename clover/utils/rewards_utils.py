from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import Tensor


@torch.no_grad()
def aesthetic_proxy_reward(images: list[Image.Image], prompts: list[str], device: torch.device | str | None = None) -> Tensor:
    """Small offline reward placeholder for preference-model based rewards."""
    rewards = []
    for image in images:
        arr = torch.from_numpy(np.asarray(image).astype("float32") / 255.0)
        mean_rgb = arr.mean(dim=(0, 1))
        brightness = arr.mean()
        saturation = (mean_rgb.max() - mean_rgb.min()).clamp_min(0.0)
        contrast = arr.std().clamp_max(0.35)
        rewards.append(0.45 * brightness + 0.35 * saturation + 0.20 * contrast)
    reward_tensor = torch.stack(rewards)
    return reward_tensor.to(device) if device is not None else reward_tensor
