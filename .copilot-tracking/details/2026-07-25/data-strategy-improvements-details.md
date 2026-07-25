<!-- markdownlint-disable-file -->
# Implementation Details: Data Strategy Improvements

## Context Reference

Sources: .copilot-tracking/research/2026-06-14/data-strategy-research.md (Current Implementation section, Actionable Next Steps section)

## Implementation Phase 1: Create Shared Prompt Registry

<!-- parallelizable: true -->

### Step 1.1: Create clover/utils/prompts.py with shared prompt definitions

Create a new module to centralize prompt definitions shared across all baselines. This eliminates the current duplication where identical clover-themed prompts appear in ddpo.py:51-56, dpok.py:51-56, and b2diffurl.py:51-56.

Files:
* clover/utils/prompts.py - New file containing DEFAULT_TRAIN_PROMPTS and DEFAULT_EVAL_PROMPTS tuples

Implementation:
```python
"""Shared prompt definitions for baseline experiments."""

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
```

Success criteria:
* File created at clover/utils/prompts.py
* DEFAULT_TRAIN_PROMPTS contains 4 clover-themed training prompts
* DEFAULT_EVAL_PROMPTS contains 4 distinct evaluation prompts with thematic overlap

Context references:
* .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 240-268) - Prompt duplication evidence

Dependencies:
* None - standalone module

### Step 1.2: Add eval_prompts separate from train_prompts

Document the rationale for separate evaluation prompts and establish the pattern for future baseline configs.

Files:
* clover/utils/prompts.py - Add module docstring explaining train/eval split philosophy

Implementation:
Add comprehensive module docstring:
```python
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
```

Success criteria:
* Module docstring explains train/eval split rationale
* Documentation clarifies when to use custom prompts vs defaults
* Future transition to manifest-based datasets is documented

Context references:
* .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 155-170) - Limitation "No train/eval splits"

Dependencies:
* Step 1.1 completion

### Step 1.3: Validate phase changes

Run lint for clover/utils/prompts.py and verify imports resolve correctly.

Validation commands:
* `python -m pylint clover/utils/prompts.py`
* `python -c "from clover.utils.prompts import DEFAULT_TRAIN_PROMPTS, DEFAULT_EVAL_PROMPTS; print(len(DEFAULT_TRAIN_PROMPTS), len(DEFAULT_EVAL_PROMPTS))"`

## Implementation Phase 2: Update Baseline Configs

<!-- parallelizable: false -->

### Step 2.1: Update DDPO config to import from shared prompts

Replace the hardcoded train_prompts tuple in ddpo.py with imports from the shared registry.

Files:
* clover/baselines/ddpo.py - Update DDPOConfig dataclass (lines 51-56)

Implementation:
1. Add import at top of file:
```python
from clover.utils.prompts import DEFAULT_EVAL_PROMPTS, DEFAULT_TRAIN_PROMPTS
```

2. Replace the train_prompts field definition:
```python
# Before:
train_prompts: tuple[str, ...] = (
    "a colorful clover field at sunrise, high detail",
    "a close-up photo of a bright green clover leaf with dew",
    "a small robot holding a clover in a clean studio photo",
    "an impressionist painting of clovers under warm sunlight",
)

# After:
train_prompts: tuple[str, ...] = DEFAULT_TRAIN_PROMPTS
eval_prompts: tuple[str, ...] = DEFAULT_EVAL_PROMPTS
```

Success criteria:
* Import statement added to ddpo.py
* train_prompts field uses DEFAULT_TRAIN_PROMPTS
* eval_prompts field added using DEFAULT_EVAL_PROMPTS
* No behavior change in existing experiments

Context references:
* .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 240-255) - DDPO config prompt duplication

Dependencies:
* Implementation Phase 1 completion

### Step 2.2: Update DPOK config to import from shared prompts

Apply the same pattern to dpok.py configuration.

Files:
* clover/baselines/dpok.py - Update DPOKConfig dataclass (lines 51-56)

Implementation:
1. Add import at top of file:
```python
from clover.utils.prompts import DEFAULT_EVAL_PROMPTS, DEFAULT_TRAIN_PROMPTS
```

2. Replace the train_prompts field definition:
```python
# Before:
train_prompts: tuple[str, ...] = (
    "a colorful clover field at sunrise, high detail",
    "a close-up photo of a bright green clover leaf with dew",
    "a small robot holding a clover in a clean studio photo",
    "an impressionist painting of clovers under warm sunlight",
)

# After:
train_prompts: tuple[str, ...] = DEFAULT_TRAIN_PROMPTS
eval_prompts: tuple[str, ...] = DEFAULT_EVAL_PROMPTS
```

Success criteria:
* Import statement added to dpok.py
* train_prompts field uses DEFAULT_TRAIN_PROMPTS
* eval_prompts field added using DEFAULT_EVAL_PROMPTS
* Config remains backward compatible

Context references:
* .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 256-261) - DPOK config prompt duplication

Dependencies:
* Implementation Phase 1 completion

### Step 2.3: Update B2-DiffuRL config to import from shared prompts

Apply the same pattern to b2diffurl.py configuration.

Files:
* clover/baselines/b2diffurl.py - Update B2DiffuRLConfig dataclass (lines 51-56)

Implementation:
1. Add import at top of file:
```python
from clover.utils.prompts import DEFAULT_EVAL_PROMPTS, DEFAULT_TRAIN_PROMPTS
```

2. Replace the train_prompts field definition:
```python
# Before:
train_prompts: tuple[str, ...] = (
    "a colorful clover field at sunrise, high detail",
    "a close-up photo of a bright green clover leaf with dew",
    "a small robot holding a clover in a clean studio photo",
    "an impressionist painting of clovers under warm sunlight",
)

# After:
train_prompts: tuple[str, ...] = DEFAULT_TRAIN_PROMPTS
eval_prompts: tuple[str, ...] = DEFAULT_EVAL_PROMPTS
```

Success criteria:
* Import statement added to b2diffurl.py
* train_prompts field uses DEFAULT_TRAIN_PROMPTS
* eval_prompts field added using DEFAULT_EVAL_PROMPTS
* Config remains backward compatible

Context references:
* .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 262-268) - B2-DiffuRL config prompt duplication

Dependencies:
* Implementation Phase 1 completion

### Step 2.4: Validate phase changes

Run lint for all modified baseline files and verify configs instantiate correctly.

Validation commands:
* `python -m pylint clover/baselines/ddpo.py clover/baselines/dpok.py clover/baselines/b2diffurl.py`
* `python -c "from clover.baselines.ddpo import DDPOConfig; from clover.baselines.dpok import DPOKConfig; from clover.baselines.b2diffurl import B2DiffuRLConfig; print('Configs OK')"`

## Implementation Phase 3: Add Configurable Reward Functions

<!-- parallelizable: true -->

### Step 3.1: Extend rewards_utils.py with reward function registry

Create a registry pattern to support multiple reward functions without hardcoding specific choices in each baseline.

Files:
* clover/utils/rewards_utils.py - Add REWARD_REGISTRY dictionary and get_reward_fn helper

Implementation:
Add after the aesthetic_proxy_reward function (after line 24):
```python
# Reward function registry
REWARD_REGISTRY: dict[str, Callable[[list[Image.Image], list[str], torch.device], Tensor]] = {
    "aesthetic": aesthetic_proxy_reward,
    # Future: "clip": clip_reward,
    # Future: "bert": bert_reward,
    # Future: "imagereward": imagereward_score,
}

def get_reward_fn(reward_type: str) -> Callable[[list[Image.Image], list[str], torch.device], Tensor]:
    """Get reward function by type name.
    
    Args:
        reward_type: Name of reward function ("aesthetic", "clip", "bert", "imagereward")
        
    Returns:
        Callable reward function matching signature (images, prompts, device) -> Tensor
        
    Raises:
        KeyError: If reward_type is not registered
    """
    if reward_type not in REWARD_REGISTRY:
        available = ", ".join(REWARD_REGISTRY.keys())
        raise KeyError(f"Unknown reward type '{reward_type}'. Available: {available}")
    return REWARD_REGISTRY[reward_type]
```

Success criteria:
* REWARD_REGISTRY dictionary created with "aesthetic" entry
* get_reward_fn helper function added with clear signature
* Function raises informative error for unknown reward types
* Comments indicate future extensibility for CLIP, BERT, ImageReward

Context references:
* .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 171-174) - Single reward function limitation

Dependencies:
* None - extends existing rewards_utils.py

### Step 3.2: Update make_reward_fn to support reward_type config parameter

Modify the make_reward_fn function in common.py to accept and use the reward_type configuration.

Files:
* clover/baselines/common.py - Update make_reward_fn function (lines 30-35)

Implementation:
Replace current implementation:
```python
# Before:
def make_reward_fn(device: torch.device) -> Callable:
    """Return the reward function."""
    from clover.utils.rewards_utils import aesthetic_proxy_reward
    return lambda images, prompts: aesthetic_proxy_reward(images, prompts, device)

# After:
def make_reward_fn(device: torch.device, reward_type: str = "aesthetic") -> Callable:
    """Return the configured reward function.
    
    Args:
        device: Torch device for reward computation
        reward_type: Type of reward function to use (default: "aesthetic")
        
    Returns:
        Callable with signature (images, prompts) -> Tensor
    """
    from clover.utils.rewards_utils import get_reward_fn
    reward_fn = get_reward_fn(reward_type)
    return lambda images, prompts: reward_fn(images, prompts, device)
```

Success criteria:
* make_reward_fn accepts reward_type parameter with default "aesthetic"
* Function uses get_reward_fn from rewards_utils
* Backward compatibility maintained through default parameter
* Clear docstring documents parameter

Context references:
* .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 93-95) - make_reward_fn current implementation

Dependencies:
* Step 3.1 completion

### Step 3.3: Add reward_type field to baseline configs

Add reward_type configuration field to each baseline's config dataclass.

Files:
* clover/baselines/ddpo.py - Add reward_type field to DDPOConfig
* clover/baselines/dpok.py - Add reward_type field to DPOKConfig
* clover/baselines/b2diffurl.py - Add reward_type field to B2DiffuRLConfig

Implementation:
For each baseline config, add field after train_prompts/eval_prompts:
```python
reward_type: str = "aesthetic"  # Options: "aesthetic" (more to come)
```

Update make_reward_fn calls in main training functions:
```python
# Before:
reward_fn = make_reward_fn(device)

# After:
reward_fn = make_reward_fn(device, config.reward_type)
```

Success criteria:
* reward_type field added to all three baseline configs
* Default value is "aesthetic" for backward compatibility
* All make_reward_fn calls updated to pass config.reward_type
* Config can be overridden via CLI (handled by existing parse_config)

Context references:
* .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 93-95) - Common utilities

Dependencies:
* Step 3.2 completion

### Step 3.4: Validate phase changes

Run lint for modified files and verify reward function selection works correctly.

Validation commands:
* `python -m pylint clover/utils/rewards_utils.py clover/baselines/common.py`
* Test reward function selection:
  ```python
  python -c "
  from clover.utils.rewards_utils import get_reward_fn
  fn = get_reward_fn('aesthetic')
  print(f'Got reward function: {fn.__name__}')
  "
  ```

## Implementation Phase 4: Update Evaluation to Use Eval Prompts

<!-- parallelizable: false -->

### Step 4.1: Update standard_eval_prompts to use config.eval_prompts when available

Modify the evaluation prompt selection logic to prefer explicit eval_prompts when defined.

Files:
* clover/utils/baseline_utils.py - Update standard_eval_prompts function (lines 640-653)

Implementation:
Replace current implementation:
```python
# Before:
def standard_eval_prompts(config: Any, limit: int = 4) -> list[str]:
    prompts = [config.prompt, *list(config.train_prompts)]
    deduped = list(dict.fromkeys(prompts))
    return deduped[:limit]

# After:
def standard_eval_prompts(config: Any, limit: int = 4) -> list[str]:
    """Get evaluation prompts from config.
    
    Prefers config.eval_prompts if defined, otherwise falls back to
    deduplicating [config.prompt, *config.train_prompts].
    
    Args:
        config: Baseline config with prompt fields
        limit: Maximum number of prompts to return
        
    Returns:
        List of evaluation prompts, up to limit
    """
    # Prefer explicit eval_prompts when available
    if hasattr(config, "eval_prompts") and config.eval_prompts:
        return list(config.eval_prompts)[:limit]
    
    # Fallback to legacy behavior for backward compatibility
    prompts = [config.prompt, *list(config.train_prompts)]
    deduped = list(dict.fromkeys(prompts))
    return deduped[:limit]
```

Success criteria:
* Function checks for config.eval_prompts attribute
* Uses eval_prompts when available
* Falls back to legacy deduplication for backward compatibility
* Maintains limit parameter behavior
* Clear docstring explains preference order

Context references:
* .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 120-127) - standard_eval_prompts current implementation

Dependencies:
* Implementation Phase 2 completion (configs have eval_prompts field)

### Step 4.2: Update baseline evaluate calls to use new eval logic

Verify that evaluation calls in each baseline use the updated standard_eval_prompts function correctly. No code changes should be needed if baselines already call standard_eval_prompts.

Files:
* clover/baselines/ddpo.py - Verify evaluate call
* clover/baselines/dpok.py - Verify evaluate call
* clover/baselines/b2diffurl.py - Verify evaluate call

Implementation:
Verify existing calls already use standard_eval_prompts:
```python
# Should already exist:
evaluate(pipe, config, device, dtype, generator_eval)
```

The evaluate function in common.py should already call standard_eval_prompts internally. Verify this is the case.

Success criteria:
* All baselines call evaluate function from common.py
* common.py evaluate function uses standard_eval_prompts
* No code changes required (verification only)

Context references:
* .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 93-95) - Common utilities

Dependencies:
* Step 4.1 completion

### Step 4.3: Validate phase changes

Run lint for modified files and verify evaluation uses correct prompt split.

Validation commands:
* `python -m pylint clover/utils/baseline_utils.py`
* Test prompt selection:
  ```python
  python -c "
  from dataclasses import dataclass
  from clover.utils.baseline_utils import standard_eval_prompts
  
  @dataclass
  class TestConfig:
      prompt: str = 'test'
      train_prompts: tuple = ('a', 'b')
      eval_prompts: tuple = ('c', 'd', 'e')
  
  config = TestConfig()
  prompts = standard_eval_prompts(config, limit=4)
  assert prompts == ['c', 'd', 'e'], f'Expected eval_prompts, got {prompts}'
  print('Eval prompt selection works correctly')
  "
  ```

## Implementation Phase 5: Validation

<!-- parallelizable: false -->

### Step 5.1: Run full project validation

Execute all validation commands for the project:
* Run all linters: `python -m pylint clover/`
* Type checking if configured: `python -m mypy clover/`
* Run smoke test for each baseline:
  ```bash
  python -m clover.baselines.ddpo --train-epochs 1 --rollouts-per-epoch 1 --output-dir outputs/test_ddpo
  python -m clover.baselines.dpok --train-epochs 1 --rollouts-per-epoch 1 --output-dir outputs/test_dpok
  python -m clover.baselines.b2diffurl --train-epochs 1 --rollouts-per-epoch 1 --output-dir outputs/test_b2diffurl
  ```

### Step 5.2: Fix minor validation issues

Iterate on lint errors, type warnings, and smoke test failures. Apply fixes directly when corrections are straightforward (import ordering, docstring formatting, type hints).

### Step 5.3: Report blocking issues

When validation failures require changes beyond minor fixes:
* Document the issues and affected files
* Provide the user with next steps and recommended investigation
* Avoid large-scale refactoring within this phase

## Dependencies

* Python 3.9+
* Existing baseline implementations
* torch, PIL, numpy (already in project)

## Success Criteria

* All baselines import from shared clover/utils/prompts.py registry
* Each baseline has separate train_prompts and eval_prompts fields
* Reward function configurable via reward_type without baseline code changes
* All tests pass, no regressions in existing experiments
* Full project validation passes cleanly
