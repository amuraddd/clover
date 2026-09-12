"""Load trained baseline checkpoints and generate images for evaluation."""

import importlib
import json
from dataclasses import dataclass, fields, replace
from typing import Any
from pathlib import Path

import torch
from diffusers import DDPMScheduler
from peft.utils import set_peft_model_state_dict
from clover.utils.baseline_utils import load_lora_pipeline, load_reference_pipeline, unwrap_unet
from clover.baselines.common import generate_eval_images

_BASELINE_TYPES = {
    "emo": ("emo", "EMOConfig"),
    "emo_v2": ("emo_v2", "EMOV2V2Config"),
    "emo_v3": ("emo_v3", "EMOV3Config"),
    "emo_v4": ("emo_v4", "EMOV2V2Config"),
    "emo_v3_old": ("emo_v3", "EMOV3Config"),
    "ddpo": ("ddpo", "DDPOConfig"),
    "b2diffurl": ("b2diffurl", "B2DiffuRLConfig"),
    "dpok": ("dpok", "DPOKConfig"),
}
_VARIANCE_MODULES = {
    "emo_v2": "emo_v2", "emo_v3": "emo_v2",
    "emo_v3_old": "emo_v2", "emo_v4": "emo_v4",
}


@dataclass
class LoadedBaseline:
    baseline: str
    pipe: Any
    config: Any
    device: torch.device
    dtype: torch.dtype
    checkpoint: Path | None
    variance_module: Any = None

    @torch.inference_mode()
    def generate(self, prompts, *, seed=123, **overrides):
        """Return RGB PIL images; override steps, guidance, dimensions or negative_prompt."""
        self.pipe.set_progress_bar_config(disable=True)
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        if not prompts or not all(isinstance(p, str) for p in prompts):
            raise ValueError("prompts must be a string or a nonempty sequence of strings")
        allowed = {"num_inference_steps", "guidance_scale", "height", "width", "negative_prompt"}
        unknown = set(overrides) - allowed
        if unknown:
            raise TypeError(f"Unsupported inference options: {sorted(unknown)}")
        config = replace(self.config, **overrides)
        factor = self.pipe.vae_scale_factor
        if any(size <= 0 or size % factor for size in (config.height, config.width)):
            raise ValueError(f"height and width must be positive multiples of {factor}")
        if config.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if self.variance_module is not None:
            images = self.variance_module._generate_emo_v2_eval_images(
                self.pipe, prompts, config, self.device, self.dtype, seed=seed
            )
        else:
            images = generate_eval_images(self.pipe, prompts, config, self.device, seed=seed)
        return [image.convert("RGB") for image in images]


def load_baseline_model(baseline, *, seed=123, output_root=None, run_dir=None,
                        device=None, dtype=None, checkpoint_file=None):
    """Load one baseline's checkpoint for inference without restoring optimizer/RNG state.

    "sd15" loads vanilla SD 1.5 with no adapter and no checkpoint.
    Other run_dir values must contain config.json and checkpoint/checkpoint.pt.
    With output_root, select seed_<seed> first, then the legacy unseeded layout.
    checkpoint_file can select archived weights while run_dir supplies config.json.
    """
    if baseline == "sd15":
        from clover.baselines.ddpo import DDPOConfig
        config = DDPOConfig(gradient_checkpointing=False, use_data_parallel=False)
        target = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        dtype = dtype or (torch.float16 if target.type == "cuda" else torch.float32)
        pipe = load_reference_pipeline(config, target, dtype)
        for component in (pipe.unet, pipe.vae, pipe.text_encoder):
            component.requires_grad_(False)
            component.eval()
        return LoadedBaseline("sd15", pipe, config, target, dtype, None)
    if baseline not in _BASELINE_TYPES:
        raise ValueError(f"Unknown baseline {baseline!r}; choose from {['sd15', *_BASELINE_TYPES]}")
    if run_dir is None:
        root = Path(output_root) if output_root is not None else Path(__file__).resolve().parents[2] / "outputs"
        baseline_dir = root / baseline
        seeded = baseline_dir / f"seed_{seed}"
        run_dir = seeded if seeded.is_dir() else baseline_dir
    run_dir = Path(run_dir)
    checkpoint = Path(checkpoint_file) if checkpoint_file is not None else run_dir / "checkpoint" / "checkpoint.pt"
    config_file = run_dir / "config.json"
    variance_file = checkpoint.parent / "variance_head.pt"
    required = [checkpoint, config_file]
    if baseline in _VARIANCE_MODULES:
        required.append(variance_file)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {baseline} artifact: {path}")

    module_name, type_name = _BASELINE_TYPES[baseline]
    config_type = getattr(importlib.import_module(f"clover.baselines.{module_name}"), type_name)
    saved = json.loads(config_file.read_text())
    accepted = {field.name for field in fields(config_type) if field.init}
    config = config_type(**{key: value for key, value in saved.items() if key in accepted})
    config = replace(config, output_dir=str(run_dir), gradient_checkpointing=False,
                     use_data_parallel=False)
    target = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dtype = dtype or (torch.float16 if target.type == "cuda" else torch.float32)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["lora_state_dict"]
    if not state:
        raise ValueError(f"Empty LoRA state: {checkpoint}")
    pipe = load_lora_pipeline(config, target, dtype, gpu_ids=[])
    variance_module = None
    if baseline in _VARIANCE_MODULES:
        variance_module = importlib.import_module(f"clover.baselines.{_VARIANCE_MODULES[baseline]}")
        variance_module._install_learned_variance_head(pipe)
        variance_module._load_variance_head(pipe, variance_file)
        pipe.scheduler = DDPMScheduler.from_config(
            pipe.scheduler.config, variance_type="learned_range", clip_sample=False
        )
    result = set_peft_model_state_dict(unwrap_unet(pipe.unet), state)
    missing = [key for key in result.missing_keys if "lora_" in key]
    if missing or result.unexpected_keys:
        raise ValueError(f"Checkpoint adapter mismatch: missing={missing}, unexpected={result.unexpected_keys}")
    for component in (pipe.unet, pipe.vae, pipe.text_encoder):
        component.requires_grad_(False)
        component.eval()
    return LoadedBaseline(baseline, pipe, config, target, dtype, checkpoint, variance_module)
