"""Sample held-out prompts from trained baseline checkpoints."""

from __future__ import annotations

import gc
import importlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
from diffusers.utils import logging as diffusers_logging
from huggingface_hub.utils import disable_progress_bars as disable_hf_progress_bars
from PIL import Image
from peft.utils import set_peft_model_state_dict
from transformers.utils import logging as transformers_logging

from clover.utils.baseline_utils import load_lora_pipeline, unwrap_unet
from clover.utils.diversity_score import _inception_outputs
from clover.utils.prompts import get_prompts


BASELINE_CONFIGS = {
    "ddpo": ("clover.baselines.ddpo", "DDPOConfig"),
    "dpok": ("clover.baselines.dpok", "DPOKConfig"),
    "md3po": ("clover.baselines.md3po", "MD3POConfig"),
    "b2diffurl": ("clover.baselines.b2diffurl", "B2DiffuRLConfig"),
}


def _suppress_huggingface_output() -> None:
    """Disable routine Hugging Face logs and tqdm progress displays."""
    diffusers_logging.set_verbosity_error()
    diffusers_logging.disable_progress_bar()
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
    disable_hf_progress_bars()


def _baseline_config(baseline: str, output_dir: Path) -> Any:
    canonical_name = baseline.removesuffix("_old")
    if canonical_name not in BASELINE_CONFIGS:
        supported = ", ".join(sorted(BASELINE_CONFIGS))
        raise ValueError(f"unsupported baseline {baseline!r}; expected one of: {supported}")
    module_name, config_name = BASELINE_CONFIGS[canonical_name]
    config_type = getattr(importlib.import_module(module_name), config_name)
    overrides: dict[str, Any] = {"output_dir": str(output_dir)}
    config_path = output_dir / "config.json"
    if config_path.is_file():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        accepted = {field.name for field in fields(config_type)}
        overrides.update({key: value for key, value in saved.items() if key in accepted})
        overrides["output_dir"] = str(output_dir)
    return config_type(**overrides)


@torch.inference_mode()
def sample_baseline_checkpoints(
    baselines: Sequence[str],
    num_images_per_prompt: int,
    *,
    base_seed: int = 10_000,
    output_root: str | Path = "outputs",
    checkpoint_paths: Mapping[str, str | Path] | None = None,
    prompt_seed: int = 123,
    device: torch.device | str | None = None,
    batch_size: int = 4,
) -> pd.DataFrame:
    """Generate multiple images per held-out prompt from baseline checkpoints.

    The same sampling seeds are reused across baselines and prompts to make
    comparisons paired and reproducible. Each baseline is loaded, sampled, and
    released before the next is loaded, so at most one model occupies the GPU.

    The result columns are seed, prompt, image, and baseline. Image values are
    in-memory RGB PIL images.
    """
    if not baselines:
        return pd.DataFrame(columns=["seed", "prompt", "image", "baseline"])
    if num_images_per_prompt <= 0:
        raise ValueError("num_images_per_prompt must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    _suppress_huggingface_output()
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if target_device.type == "cuda" and target_device.index not in (None, 0):
        raise ValueError("evaluation is restricted to one visible GPU (cuda or cuda:0)")
    dtype = torch.float16 if target_device.type == "cuda" else torch.float32
    _, eval_prompts = get_prompts(seed=prompt_seed, save=False)
    sampling_seeds = [base_seed + index for index in range(num_images_per_prompt)]
    output_root = Path(output_root)
    checkpoint_paths = checkpoint_paths or {}
    records: list[dict[str, Any]] = []

    for baseline in baselines:
        output_dir = output_root / baseline
        checkpoint_path = Path(
            checkpoint_paths.get(baseline, output_dir / "checkpoint" / "checkpoint.pt")
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found for {baseline}: {checkpoint_path}")

        config = _baseline_config(baseline, output_dir)
        pipe = load_lora_pipeline(config, target_device, dtype, gpu_ids=[])
        pipe.set_progress_bar_config(disable=True)
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "lora_state_dict" not in checkpoint:
                raise KeyError(f"checkpoint has no lora_state_dict: {checkpoint_path}")
            set_peft_model_state_dict(unwrap_unet(pipe.unet), checkpoint["lora_state_dict"])
            pipe.unet.eval()

            for prompt in eval_prompts:
                for start in range(0, len(sampling_seeds), batch_size):
                    seeds = sampling_seeds[start : start + batch_size]
                    generators = [
                        torch.Generator(device=target_device).manual_seed(seed) for seed in seeds
                    ]
                    images = pipe(
                        [prompt] * len(seeds),
                        negative_prompt=[config.negative_prompt] * len(seeds),
                        height=config.height,
                        width=config.width,
                        num_inference_steps=config.num_inference_steps,
                        guidance_scale=config.guidance_scale,
                        generator=generators,
                    ).images
                    for seed, image in zip(seeds, images):
                        if not isinstance(image, Image.Image):
                            raise TypeError("diffusion pipeline returned a non-PIL image")
                        records.append(
                            {
                                "seed": seed,
                                "prompt": prompt,
                                "image": image.convert("RGB"),
                                "baseline": baseline,
                            }
                        )
        finally:
            del pipe
            gc.collect()
            if target_device.type == "cuda":
                torch.cuda.empty_cache()

    return pd.DataFrame(records, columns=["seed", "prompt", "image", "baseline"])


def prompt_baseline_fid_scores(
    dataframe: pd.DataFrame,
    *,
    device: torch.device | str | None = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Calculate mean pairwise singleton FID for each prompt and baseline.

    Since the sampled-image dataframe has no separate real/reference set, this
    measures within-group spread: Inception features are calculated once, then
    the squared feature distance (singleton FID) is averaged over every unique
    pair of images. Larger scores indicate greater feature-space diversity.
    """
    required = {"prompt", "image", "baseline"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"dataframe is missing required columns: {sorted(missing)}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    result_columns = ["prompt", "FID score", "baseline"]
    if dataframe.empty:
        return pd.DataFrame(columns=result_columns)

    rows = []
    for (baseline, prompt), group in dataframe.groupby(
        ["baseline", "prompt"], sort=False, dropna=False
    ):
        images = group["image"].tolist()
        if len(images) < 2:
            raise ValueError(
                f"at least two images are required for baseline={baseline!r}, prompt={prompt!r}"
            )
        if not all(isinstance(image, Image.Image) for image in images):
            raise TypeError("the image column must contain PIL images")
        features = _inception_outputs(
            images, device=device, batch_size=batch_size, features=True
        ).double()
        score = float(torch.pdist(features, p=2).square().mean())
        rows.append({"prompt": prompt, "FID score": score, "baseline": baseline})

    return pd.DataFrame(rows, columns=result_columns)
