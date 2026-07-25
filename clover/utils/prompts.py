"""Shared prompt definitions for baseline experiments.

This module provides default training and evaluation prompts for diffusion model
baselines. The separation between train and eval prompts enables better evaluation
methodology while maintaining thematic consistency.

Design rationale:
- Training prompts are used during RL rollout collection
- Evaluation prompts are held-out and used only for evaluation
- Eval prompts have thematic overlap with training but are distinct
- This prevents eval reward inflation from memorization

When to use custom prompts:
- Override config.train_prompts and config.eval_prompts in baseline configs
- For manifest-based datasets, this module is superseded by data/manifests/
"""

# Default training prompts shared across baselines
DEFAULT_TRAIN_PROMPTS: tuple[str, ...] = (
    "a colorful clover field at sunrise, high detail",
    "a close-up photo of a bright green clover leaf with dew",
    "a small robot holding a clover in a clean studio photo",
    "an impressionist painting of clovers under warm sunlight",
)

# Default evaluation prompts (separate from training for better evaluation methodology)
DEFAULT_EVAL_PROMPTS: tuple[str, ...] = (
    "a colorful clover field at sunrise, high detail",
    "a photorealistic clover close-up with water droplets",
    "a robot holding a four-leaf clover, studio lighting",
    "an oil painting of clovers in golden hour light",
)
