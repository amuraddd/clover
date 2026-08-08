"""Inception-based quality and distribution metrics for generated images."""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.linalg import sqrtm
from torch import Tensor, nn

ImageBatch = Sequence[Image.Image] | Tensor


def _device(device: torch.device | str | None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _image_tensor(images: ImageBatch) -> Tensor:
    if torch.is_tensor(images):
        batch = images.detach()
        if batch.ndim == 3:
            batch = batch.unsqueeze(0)
        if batch.ndim != 4:
            raise ValueError("image tensors must have shape [N, C, H, W]")
        if batch.shape[1] not in (1, 3, 4) and batch.shape[-1] in (1, 3, 4):
            batch = batch.permute(0, 3, 1, 2)
        batch = batch[:, :3].float()
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        if images.dtype == torch.uint8 or float(batch.max()) > 1.0:
            batch = batch / 255.0
    else:
        converted = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise TypeError("images must contain PIL images or be a tensor")
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            converted.append(torch.from_numpy(array).permute(2, 0, 1))
        if not converted:
            raise ValueError("images must not be empty")
        batch = torch.stack(converted).float() / 255.0
    if batch.shape[0] == 0:
        raise ValueError("images must not be empty")
    return batch.clamp(0, 1)


@lru_cache(maxsize=4)
def _inception_model(device_string: str, features: bool) -> nn.Module:
    from torchvision.models import Inception_V3_Weights, inception_v3

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    if features:
        model.fc = nn.Identity()
    return model.eval().to(device_string)


@torch.no_grad()
def _inception_outputs(
    images: ImageBatch,
    *,
    device: torch.device | str | None = None,
    batch_size: int = 32,
    features: bool,
    model: nn.Module | None = None,
) -> Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    target_device = _device(device)
    network = model or _inception_model(str(target_device), features)
    network.eval()
    batch = _image_tensor(images)
    mean = torch.tensor((0.485, 0.456, 0.406), device=target_device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=target_device).view(1, 3, 1, 1)
    outputs = []
    for chunk in batch.split(batch_size):
        chunk = F.interpolate(chunk.to(target_device), (299, 299), mode="bilinear", align_corners=False)
        output = network((chunk - mean) / std)
        if hasattr(output, "logits"):
            output = output.logits
        outputs.append(output.float().flatten(1).cpu())
    return torch.cat(outputs)


def rollout_fid_scores(
    ground_truth_images: ImageBatch,
    candidate_images: ImageBatch,
    comparison_mask: Tensor | None = None,
    *,
    device: torch.device | str | None = None,
    batch_size: int = 32,
    feature_model: nn.Module | None = None,
) -> Tensor:
    """Return normalized singleton-FID scores for replay candidates.

    Each candidate is compared with every current-iteration ground-truth image.
    Scores are min-max normalized over valid comparisons and the lowest valid
    score is returned per candidate. Invalid candidates receive infinity.
    """
    truth = _inception_outputs(
        ground_truth_images, device=device, batch_size=batch_size, features=True, model=feature_model
    )
    candidates = _inception_outputs(
        candidate_images, device=device, batch_size=batch_size, features=True, model=feature_model
    )
    distances = torch.cdist(candidates, truth).square() / truth.shape[1]
    if comparison_mask is None:
        comparison_mask = torch.ones_like(distances, dtype=torch.bool)
    else:
        comparison_mask = comparison_mask.detach().cpu().bool()
        if comparison_mask.shape != distances.shape:
            raise ValueError(f"comparison_mask must have shape {tuple(distances.shape)}")
    valid = distances[comparison_mask]
    if valid.numel() == 0:
        return torch.full((candidates.shape[0],), float("inf"))
    span = valid.max() - valid.min()
    normalized = torch.zeros_like(distances) if float(span) == 0.0 else (distances - valid.min()) / span
    normalized = normalized.masked_fill(~comparison_mask, float("inf"))
    return normalized.min(dim=1).values


def inception_score(
    images: ImageBatch,
    *,
    splits: int = 10,
    device: torch.device | str | None = None,
    batch_size: int = 32,
    classifier: nn.Module | None = None,
) -> tuple[float, float]:
    """Calculate the mean and standard deviation of the Inception Score."""
    logits = _inception_outputs(
        images, device=device, batch_size=batch_size, features=False, model=classifier
    )
    if splits <= 0 or splits > logits.shape[0]:
        raise ValueError("splits must be between one and the number of images")
    probabilities = logits.softmax(dim=1)
    scores = []
    for split in torch.tensor_split(probabilities, splits):
        marginal = split.mean(dim=0, keepdim=True)
        kl = split * (split.clamp_min(1e-12).log() - marginal.clamp_min(1e-12).log())
        scores.append(kl.sum(dim=1).mean().exp())
    values = torch.stack(scores)
    return float(values.mean()), float(values.std(unbiased=False))


def frechet_inception_distance(
    ground_truth_images: ImageBatch,
    generated_images: ImageBatch,
    *,
    device: torch.device | str | None = None,
    batch_size: int = 32,
    feature_model: nn.Module | None = None,
) -> float:
    """Calculate dataset-level FID between reference and generated images."""
    real = _inception_outputs(
        ground_truth_images, device=device, batch_size=batch_size, features=True, model=feature_model
    ).numpy().astype(np.float64)
    generated = _inception_outputs(
        generated_images, device=device, batch_size=batch_size, features=True, model=feature_model
    ).numpy().astype(np.float64)
    if len(real) < 2 or len(generated) < 2:
        raise ValueError("FID requires at least two images in each set")
    real_mean, generated_mean = real.mean(0), generated.mean(0)
    real_cov = np.cov(real, rowvar=False)
    generated_cov = np.cov(generated, rowvar=False)
    covariance_mean = sqrtm(real_cov @ generated_cov)
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    difference = real_mean - generated_mean
    score = difference @ difference + np.trace(real_cov + generated_cov - 2 * covariance_mean)
    return float(max(score, 0.0))
