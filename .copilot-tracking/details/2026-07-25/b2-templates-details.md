<!-- markdownlint-disable-file -->
# Implementation Details: B2 Prompt Templates

## Context Reference

Sources: .copilot-tracking/research/2026-07-25/b2-templates-implementation-research.md, .copilot-tracking/research/subagents/2026-07-25/b2-template-specs.md

## Implementation Phase 1: Template 1 Generator and Splits

<!-- parallelizable: true -->

### Step 1.1: Add Template 1 constants and generator function

Add to clover/utils/prompts.py after existing DEFAULT_EVAL_PROMPTS:

```python
# B2-DiffuRL Template 1: Animal-Activity Patterns
# Source: B2-DiffuRL CVPR 2025, Appendix H
# Overlap: ddpo_b2_shared - canonical DDPO/B2-DiffuRL overlap
# Pattern: a(n) [animal] [activity]
# Semantic tag: activity

TEMPLATE_1_ANIMALS: tuple[str, ...] = (
    "dog", "cat", "elephant", "rabbit", "fox", "lion", "tiger", "bear",
    "monkey", "giraffe", "zebra", "horse", "cow", "sheep", "pig", "goat",
    "chicken", "duck", "penguin", "owl", "eagle", "parrot", "flamingo",
    "turtle", "frog", "fish", "dolphin", "whale", "shark", "octopus",
    "snake", "lizard", "crocodile", "ant", "bee", "butterfly", "spider",
    "squirrel", "mouse", "rat", "hedgehog", "raccoon", "deer", "moose",
)  # Total: 45 animals (matches DDPO baseline)

TEMPLATE_1_ACTIVITIES: tuple[str, ...] = (
    "riding a bike",
    "playing chess",
    "washing dishes",
)

def generate_template_1_prompts() -> tuple[str, ...]:
    """Generate all 135 Template 1 prompts (45 animals × 3 activities).
    
    Returns Cartesian product in deterministic order.
    """
    prompts = []
    for animal in TEMPLATE_1_ANIMALS:
        for activity in TEMPLATE_1_ACTIVITIES:
            article = "an" if animal[0] in "aeiou" else "a"
            prompts.append(f"{article} {animal} {activity}")
    return tuple(prompts)

# Generate all 135 Template 1 prompts
TEMPLATE_1_ALL_PROMPTS = generate_template_1_prompts()
```

Files:
* clover/utils/prompts.py - Add Template 1 constants, generator function, and generated prompt tuple

Discrepancy references:
* None - follows research recommendations exactly

Success criteria:
* TEMPLATE_1_ANIMALS has 45 elements
* TEMPLATE_1_ACTIVITIES has 3 elements
* TEMPLATE_1_ALL_PROMPTS has 135 elements (45 × 3)
* Generator function produces deterministic Cartesian product with correct article usage

Context references:
* .copilot-tracking/research/subagents/2026-07-25/b2-template-specs.md (Lines 15-60) - Template 1 specification

Dependencies:
* Existing clover/utils/prompts.py structure

### Step 1.2: Define Template 1 train/eval splits

Add to clover/utils/prompts.py after TEMPLATE_1_ALL_PROMPTS:

```python
# Template 1 splits (90 train / 45 eval based on B2-DiffuRL paper ratios)
# Train: first 30 animals × 3 activities = 90 prompts
# Eval: last 15 animals × 3 activities = 45 prompts
# This split preserves animal-level separation (no animal appears in both splits)

TEMPLATE_1_TRAIN_PROMPTS = TEMPLATE_1_ALL_PROMPTS[:90]  # First 30 animals
TEMPLATE_1_EVAL_PROMPTS = TEMPLATE_1_ALL_PROMPTS[90:]   # Last 15 animals
```

Files:
* clover/utils/prompts.py - Add Template 1 split definitions

Success criteria:
* TEMPLATE_1_TRAIN_PROMPTS has 90 elements
* TEMPLATE_1_EVAL_PROMPTS has 45 elements
* No overlap between train and eval prompts
* Split preserves animal-level separation (first 30 animals in train, last 15 in eval)

Context references:
* .copilot-tracking/research/2026-07-25/b2-templates-implementation-research.md (Lines 285-295) - Split strategy from B2-DiffuRL paper

Dependencies:
* Step 1.1 completion (TEMPLATE_1_ALL_PROMPTS must exist)

### Step 1.3: Add docstrings and provenance comments

Enhance the Template 1 section with comprehensive documentation:

```python
"""
B2-DiffuRL Template 1: Animal-Activity Patterns

This template implements the canonical DDPO/B2-DiffuRL overlap slice (ddpo_b2_shared).
It provides 135 prompts combining 45 animals with 3 activities, testing compositional
generation and action understanding.

Source: B2-DiffuRL CVPR 2025, Appendix H
Pattern: a(n) [animal] [activity]
Semantic coverage: activity, compositional
Split strategy: Animal-level separation (first 30 animals train, last 15 eval)
"""
```

Files:
* clover/utils/prompts.py - Add comprehensive docstring above Template 1 section

Success criteria:
* Docstring documents pattern, source, semantic coverage, and split strategy
* DDPO overlap (ddpo_b2_shared) clearly identified
* Usage examples included

Dependencies:
* Steps 1.1-1.2 completion

## Implementation Phase 2: Templates 2 and 3 Constants

<!-- parallelizable: false -->

### Step 2.1: Extract Template 2 and 3 prompts from B2-DiffuRL Appendix H

Access B2-DiffuRL CVPR 2025 paper and extract complete prompt lists:

* URL: https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_Towards_Better_Alignment_Training_Diffusion_Models_with_Reinforcement_Learning_Against_CVPR_2025_paper.pdf
* Navigate to Appendix H
* Extract all 40 Template 2 prompts (color-attribute pattern)
* Extract all 40 Template 3 prompts (spatial-relational pattern)
* Document train/eval split assignments per template

Files:
* N/A - external document extraction

Success criteria:
* Complete list of 40 Template 2 prompts extracted
* Complete list of 40 Template 3 prompts extracted
* Train/eval split assignments documented
* Prompts verified against paper's pattern specifications

Context references:
* .copilot-tracking/research/subagents/2026-07-25/b2-template-specs.md (Lines 90-140) - Template 2 and 3 specifications

Dependencies:
* Access to B2-DiffuRL CVPR 2025 paper

### Step 2.2: Add Template 2 constants and splits

Add to clover/utils/prompts.py after Template 1 section:

```python
"""
B2-DiffuRL Template 2: Color-Attribute Patterns

This template tests attribute binding with 40 prompts combining colors with fruits/vegetables.
Generated via GPT-4 assistance per B2-DiffuRL methodology.

Source: B2-DiffuRL CVPR 2025, Appendix H
Pattern: [color] [fruit/vegetable]
Semantic coverage: attribute, color
"""

# B2-DiffuRL Template 2: Color-Attribute Patterns
TEMPLATE_2_PROMPTS: tuple[str, ...] = (
    # Extract from Appendix H
    "red apple",
    "green grape",
    # ... remaining 38 prompts from extraction
)

TEMPLATE_2_TRAIN_PROMPTS = TEMPLATE_2_PROMPTS[:30]  # Adjust based on paper split
TEMPLATE_2_EVAL_PROMPTS = TEMPLATE_2_PROMPTS[30:]
```

Files:
* clover/utils/prompts.py - Add Template 2 constants and splits

Discrepancy references:
* Depends on successful Appendix H extraction (Step 2.1)

Success criteria:
* TEMPLATE_2_PROMPTS has 40 elements
* All prompts follow [color] [fruit/vegetable] pattern
* Train/eval split matches B2-DiffuRL paper specification
* Docstring documents generation method (GPT-4 assisted)

Dependencies:
* Step 2.1 completion (prompts extracted from Appendix H)

### Step 2.3: Add Template 3 constants and splits

Add to clover/utils/prompts.py after Template 2 section:

```python
"""
B2-DiffuRL Template 3: Spatial-Relational Patterns

This template tests spatial reasoning with 40 prompts combining objects via spatial predicates.
Sourced from Visual Relation Dataset annotations.

Source: B2-DiffuRL CVPR 2025, Appendix H (Visual Relation Dataset)
Pattern: [object_1] [predicate] [object_2]
Semantic coverage: relation, spatial
"""

# B2-DiffuRL Template 3: Spatial-Relational Patterns
TEMPLATE_3_PROMPTS: tuple[str, ...] = (
    # Extract from Appendix H
    "a cat to the left of a couch",
    "a dog on the moon",
    # ... remaining 38 prompts from extraction
)

TEMPLATE_3_TRAIN_PROMPTS = TEMPLATE_3_PROMPTS[:30]  # Adjust based on paper split
TEMPLATE_3_EVAL_PROMPTS = TEMPLATE_3_PROMPTS[30:]
```

Files:
* clover/utils/prompts.py - Add Template 3 constants and splits

Success criteria:
* TEMPLATE_3_PROMPTS has 40 elements
* All prompts follow [object_1] [predicate] [object_2] pattern
* Train/eval split matches B2-DiffuRL paper specification
* Docstring documents source dataset (Visual Relation Dataset)

Dependencies:
* Step 2.1 completion (prompts extracted from Appendix H)

## Implementation Phase 3: Unified B2 Prompt Sets

<!-- parallelizable: true -->

### Step 3.1: Create combined B2 train and eval prompt sets

Add to clover/utils/prompts.py after Template 3 section:

```python
"""
Unified B2 Prompt Sets

Combined prompt sets spanning all three B2-DiffuRL template families.
Use these for baselines requiring complete semantic coverage across
activities, attributes, and spatial relations.

Total: 215 prompts (135 + 40 + 40)
Train: ~150 prompts
Eval: ~65 prompts
"""

# Combined B2 template sets for baselines that want all templates
B2_FULL_TRAIN_PROMPTS = (
    *TEMPLATE_1_TRAIN_PROMPTS,
    *TEMPLATE_2_TRAIN_PROMPTS,
    *TEMPLATE_3_TRAIN_PROMPTS,
)  # ~150 train prompts total

B2_FULL_EVAL_PROMPTS = (
    *TEMPLATE_1_EVAL_PROMPTS,
    *TEMPLATE_2_EVAL_PROMPTS,
    *TEMPLATE_3_EVAL_PROMPTS,
)  # ~65 eval prompts total
```

Files:
* clover/utils/prompts.py - Add unified B2 prompt sets

Success criteria:
* B2_FULL_TRAIN_PROMPTS has ~150 elements (sum of all template train sets)
* B2_FULL_EVAL_PROMPTS has ~65 elements (sum of all template eval sets)
* No overlap between train and eval sets
* Docstring documents total counts and semantic coverage

Dependencies:
* Phase 2 completion (all three template train/eval splits must exist)

### Step 3.2: Update module docstring with B2 template usage

Update the module-level docstring at top of clover/utils/prompts.py:

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

B2-DiffuRL Templates:
- Template 1 (activity): 135 prompts - animal-activity patterns, DDPO overlap
- Template 2 (attribute): 40 prompts - color-attribute patterns
- Template 3 (relation): 40 prompts - spatial-relational patterns
- Use TEMPLATE_1_*, TEMPLATE_2_*, TEMPLATE_3_* for individual templates
- Use B2_FULL_* for complete semantic coverage across all templates

Cross-baseline comparison:
- Import from this module to guarantee identical prompt sequences
- Shared imports ensure deterministic cross-baseline evaluation
- All baselines see same prompts in same order

When to use custom prompts:
- Override config.train_prompts and config.eval_prompts in baseline configs
- For manifest-based datasets, this module is superseded by data/manifests/

Migration path:
- Current config-driven approach sufficient for <300 prompts
- Transition to manifest-based when multi-scorer evaluation becomes necessary
- See data strategy research for transition criteria
"""
```

Files:
* clover/utils/prompts.py - Update module docstring

Success criteria:
* Docstring documents all three B2 templates
* Usage patterns explained (individual templates vs combined)
* Cross-baseline comparison guidance included
* Migration path to manifests documented

Dependencies:
* Step 3.1 completion (unified sets referenced in docstring)

## Implementation Phase 4: Baseline Integration and Testing

<!-- parallelizable: false -->

### Step 4.1: Test Template 1 with one baseline

Create a test script to verify Template 1 implementation:

```python
# test_template_1.py
from clover.utils.prompts import (
    TEMPLATE_1_TRAIN_PROMPTS,
    TEMPLATE_1_EVAL_PROMPTS,
    TEMPLATE_1_ALL_PROMPTS,
)

# Verify counts
assert len(TEMPLATE_1_ALL_PROMPTS) == 135, f"Expected 135 prompts, got {len(TEMPLATE_1_ALL_PROMPTS)}"
assert len(TEMPLATE_1_TRAIN_PROMPTS) == 90, f"Expected 90 train prompts, got {len(TEMPLATE_1_TRAIN_PROMPTS)}"
assert len(TEMPLATE_1_EVAL_PROMPTS) == 45, f"Expected 45 eval prompts, got {len(TEMPLATE_1_EVAL_PROMPTS)}"

# Verify no overlap
train_set = set(TEMPLATE_1_TRAIN_PROMPTS)
eval_set = set(TEMPLATE_1_EVAL_PROMPTS)
assert len(train_set & eval_set) == 0, "Train and eval prompts overlap!"

# Verify pattern
for prompt in TEMPLATE_1_ALL_PROMPTS[:5]:
    print(f"Sample prompt: {prompt}")
    assert " " in prompt, "Prompts should contain spaces"
    assert prompt.startswith(("a ", "an ")), "Prompts should start with article"

print("✓ Template 1 validation passed!")
```

Run test: `python test_template_1.py`

Files:
* test_template_1.py - Temporary test script (can be deleted after validation)

Success criteria:
* All assertions pass
* Sample prompts display correctly
* Counts match specifications (135 total, 90 train, 45 eval)
* No overlap between train and eval sets

Context references:
* .copilot-tracking/research/2026-07-25/b2-templates-implementation-research.md (Lines 385-410) - Testing methodology

Dependencies:
* Phase 1 completion (Template 1 fully implemented)

### Step 4.2: Verify cross-baseline determinism

Test that all baselines can import identical prompt sequences:

```python
# test_cross_baseline.py
from clover.utils.prompts import B2_FULL_TRAIN_PROMPTS, B2_FULL_EVAL_PROMPTS

# Simulate baseline imports
ddpo_train = B2_FULL_TRAIN_PROMPTS
dpok_train = B2_FULL_TRAIN_PROMPTS
b2_train = B2_FULL_TRAIN_PROMPTS

# Verify identity (not just equality)
assert ddpo_train is dpok_train is b2_train, "Baselines should share same tuple object"
assert id(ddpo_train) == id(dpok_train) == id(b2_train), "Memory addresses should be identical"

# Verify deterministic ordering
for i, prompt in enumerate(B2_FULL_TRAIN_PROMPTS[:5]):
    print(f"Prompt {i}: {prompt}")

print("✓ Cross-baseline determinism verified!")
```

Run test: `python test_cross_baseline.py`

Files:
* test_cross_baseline.py - Temporary test script

Success criteria:
* All baselines reference same tuple object (not copies)
* Memory addresses identical across imports
* Deterministic ordering verified

Dependencies:
* Phase 3 completion (B2_FULL_* sets implemented)

### Step 4.3: Document usage patterns and examples

Create usage examples in module docstring or separate examples file:

```python
# Example 1: Template 1 only (DDPO overlap focus)
from clover.utils.prompts import TEMPLATE_1_TRAIN_PROMPTS, TEMPLATE_1_EVAL_PROMPTS

config.train_prompts = TEMPLATE_1_TRAIN_PROMPTS  # 90 prompts
config.eval_prompts = TEMPLATE_1_EVAL_PROMPTS    # 45 prompts

# Example 2: Full B2 coverage
from clover.utils.prompts import B2_FULL_TRAIN_PROMPTS, B2_FULL_EVAL_PROMPTS

config.train_prompts = B2_FULL_TRAIN_PROMPTS  # ~150 prompts
config.eval_prompts = B2_FULL_EVAL_PROMPTS    # ~65 prompts

# Example 3: Custom template mix
from clover.utils.prompts import (
    TEMPLATE_1_TRAIN_PROMPTS,
    TEMPLATE_3_TRAIN_PROMPTS,
    TEMPLATE_1_EVAL_PROMPTS,
    TEMPLATE_3_EVAL_PROMPTS,
)

# Activities + spatial relations (skip attributes)
CUSTOM_TRAIN = (*TEMPLATE_1_TRAIN_PROMPTS, *TEMPLATE_3_TRAIN_PROMPTS)
CUSTOM_EVAL = (*TEMPLATE_1_EVAL_PROMPTS, *TEMPLATE_3_EVAL_PROMPTS)
```

Files:
* clover/utils/prompts.py - Add usage examples to module docstring or as comments

Success criteria:
* Three usage patterns documented (Template 1 only, full B2, custom mix)
* Examples show correct import syntax
* Cross-baseline comparison guidance included

Dependencies:
* All previous steps completion

## Dependencies

* Python 3.8+ (existing project requirement)
* Access to B2-DiffuRL CVPR 2025 paper Appendix H

## Success Criteria

* All three B2 template families implemented with train/eval splits
* Template 1 preserves DDPO overlap for reproducibility
* Cross-baseline determinism verified through shared imports
* Implementation maintains fast iteration speed
* Total prompt count = 215 (135 + 40 + 40) with ~150 train / ~65 eval split
* Migration path to manifest-based architecture documented
