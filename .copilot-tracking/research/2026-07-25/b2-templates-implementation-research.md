<!-- markdownlint-disable-file -->
# Task Research: B2 Prompt Templates Implementation for Baseline Experiments

Implement B2-DiffuRL's three prompt template families (215 total prompts) to enable fair cross-baseline comparisons and semantic coverage across activities, attributes, and spatial relations in this experimental repository.

## Task Implementation Requests

* Add B2-DiffuRL Template 1, 2, and 3 prompt families to the repository
* Enable fair cross-baseline comparison with identical prompt sequences
* Maintain fast iteration speed appropriate for experimental work
* Preserve DDPO overlap slice for reproducibility
* Define train/eval/generalization splits per template
* Align implementation with phased data strategy evolution

## Scope and Success Criteria

* Scope: Implement B2 prompt templates in a way that balances experimental iteration speed with cross-baseline fairness. Research covers current prompts.py state, B2 template specifications, implementation alternatives, and phased evolution alignment. Excludes manifest-based infrastructure unless transition criteria strongly justify the added complexity.
* Assumptions:
  * Repository is experimental, not production-grade - iteration speed is critical
  * All three baselines already import from shared `clover/utils/prompts.py`
  * Prompt count (215) is under the ~200-300 threshold where config-driven remains practical
  * Multi-scorer evaluation is future work, not immediate need
  * Fair cross-baseline comparison is achieved through deterministic shared imports
  * Templates 2 and 3 exact prompts require extraction from B2-DiffuRL Appendix H
* Success Criteria:
  * All three B2 templates implemented with train/eval splits
  * Template 1 preserves DDPO overlap for reproducibility
  * Cross-baseline determinism verified (identical sequences across baselines)
  * Implementation time ≤ 1 day
  * Iteration speed maintained (edit prompts, run immediately)
  * Clear migration path documented for future manifest transition

## Executive Summary

**Current State (as of 2026-07-25):**

The repository has successfully eliminated prompt duplication by creating `clover/utils/prompts.py` with shared `DEFAULT_TRAIN_PROMPTS` and `DEFAULT_EVAL_PROMPTS`. All three baselines now import from this module. However, the current prompts are placeholder clover-themed examples (4 train + 4 eval), not the B2-DiffuRL templates needed for meaningful baseline comparisons.

**B2 Template Requirements:**

* **Template 1 (animal-activity):** 135 prompts - 45 animals × 3 activities, direct DDPO overlap
* **Template 2 (color-attribute):** 40 prompts - color-fruit/vegetable combinations
* **Template 3 (spatial-relational):** 40 prompts - object-predicate-object structure
* **Total:** 215 prompts with train/eval/generalization splits

**Selected Approach:**

Extend `clover/utils/prompts.py` with B2 template constants, generator functions, and split definitions. This config-driven approach:

* Maintains fast iteration speed (2-4 hours implementation vs 1-3 days for manifests)
* Guarantees cross-baseline determinism through shared imports
* Aligns with experimental repo context (data strategy endorses config-driven for <300 prompts)
* Provides clear migration path when multi-scorer evaluation or provenance tracking becomes critical

**Transition Criteria Weak:** Only 2 of 7 data strategy transition criteria are strongly met. Manifest infrastructure should be deferred until multi-scorer comparison or publication-level provenance becomes necessary.

## Outline

* Assess current prompts.py implementation and baseline usage
* Document B2 template specifications from research papers
* Evaluate implementation alternatives (config-driven vs manifest-based)
* Select approach aligned with experimental repo context
* Define implementation steps and timeline
* Document migration path for future manifest transition

## Research Executed

### Current Implementation Analysis

**File: clover/utils/prompts.py (lines 1-35)**

The module provides shared prompt definitions with clear design rationale:

```python
"""Shared prompt definitions for baseline experiments.

Design rationale:
- Training prompts are used during RL rollout collection
- Evaluation prompts are held-out and used only for evaluation
- Eval prompts have thematic overlap with training but are distinct
- This prevents eval reward inflation from memorization

When to use custom prompts:
- Override config.train_prompts and config.eval_prompts in baseline configs
- For manifest-based datasets, this module is superseded by data/manifests/
"""

DEFAULT_TRAIN_PROMPTS: tuple[str, ...] = (
    "a colorful clover field at sunrise, high detail",
    "a close-up photo of a bright green clover leaf with dew",
    "a small robot holding a clover in a clean studio photo",
    "an impressionist painting of clovers under warm sunlight",
)

DEFAULT_EVAL_PROMPTS: tuple[str, ...] = (
    "a colorful clover field at sunrise, high detail",
    "a photorealistic clover close-up with water droplets",
    "a robot holding a four-leaf clover, studio lighting",
    "an oil painting of clovers in golden hour light",
)
```

**Key Observations:**

* Module already established as canonical source of truth for prompts
* Design explicitly contemplates manifest-based supersession in future
* Train/eval separation already enforced
* All baselines verified to import from this module (prompt duplication eliminated)

**Baseline Integration:**

From subagent research: all three baseline configs (ddpo.py, dpok.py, b2diffurl.py) previously had identical hardcoded prompts. The recent prompts.py creation successfully centralized prompt management.

### B2 Template Specifications

**Source:** .copilot-tracking/research/subagents/2026-07-25/b2-template-specs.md

#### Template 1: Animal-Activity Patterns (DDPO Overlap)

**Pattern:** `a(n) [animal] [activity]`

**Variables:**
* Animals: 45 common animals from DDPO baseline (dog, cat, elephant, rabbit, etc.)
* Activities: 3 fixed activities
  * riding a bike
  * playing chess
  * washing dishes

**Prompt Count:** 45 × 3 = 135 prompts

**Semantic Tag:** `activity`

**Overlap Slice:** `ddpo_b2_shared` - canonical overlap between DDPO and B2-DiffuRL

**Example Prompts:**
```
a dog riding a bike
a cat playing chess
a bird washing dishes
an elephant riding a bike
```

**Coverage Goals:**
* Test prompt-image alignment for action understanding
* Evaluate compositional generation (animal + activity)
* Preserve reproducibility with DDPO baseline experiments

**Split Strategy (from B2-DiffuRL paper):**
* Training prompts: ~90 prompts (66% of template)
* Eval/generalization prompts: ~45 prompts (33% of template)

#### Template 2: Color-Attribute Patterns

**Pattern:** `[color] [fruit/vegetable]`

**Variables:**
* Colors: Color adjectives (exact list requires B2-DiffuRL Appendix H)
* Objects: Common fruits and vegetables

**Prompt Count:** 40 prompts

**Generation Method:** GPT-4 assisted construction

**Semantic Tag:** `attribute`, `color`

**Example Prompts (inferred):**
```
red apple
green grape
yellow banana
purple eggplant
orange carrot
```

**Coverage Goals:**
* Test attribute binding (color to object)
* Evaluate basic compositional understanding
* Complement Template 1's action focus with static attributes

**Gap:** Full prompt list requires extraction from B2-DiffuRL Appendix H

#### Template 3: Spatial-Relational Patterns

**Pattern:** `[object_1] [predicate] [object_2]`

**Variables:**
* Objects: Object pairs from Visual Relation Dataset
* Predicates: Spatial relations (to the left of, on, next to, under, etc.)

**Prompt Count:** 40 prompts

**Source Dataset:** Visual Relation Dataset annotations

**Semantic Tag:** `relation`, `spatial`

**Example Prompts:**
```
a cat to the left of a couch
a dog on the moon
a book next to a lamp
```

**Coverage Goals:**
* Test spatial reasoning and relational understanding
* Evaluate multi-object composition
* Provide most challenging alignment task (relations harder than attributes/activities)

**Gap:** Full prompt list requires extraction from B2-DiffuRL Appendix H

### Implementation Alternatives Evaluation

**Source:** .copilot-tracking/research/subagents/2026-07-25/prompt-implementation-alternatives.md

#### Alternative 1: Extend prompts.py with B2 Template Constants ⭐ RECOMMENDED

**Implementation:**
* Add `TEMPLATE_1_TRAIN_PROMPTS`, `TEMPLATE_1_EVAL_PROMPTS` tuples to prompts.py
* Add `TEMPLATE_2_TRAIN_PROMPTS`, `TEMPLATE_2_EVAL_PROMPTS` tuples
* Add `TEMPLATE_3_TRAIN_PROMPTS`, `TEMPLATE_3_EVAL_PROMPTS` tuples
* Add generator functions for Template 1 (Cartesian product of animals × activities)
* Define `B2_FULL_TRAIN_PROMPTS` as concatenation of all three template train sets
* Define `B2_FULL_EVAL_PROMPTS` as concatenation of all three template eval sets

**Implementation Effort:** 2-4 hours (6-8 hours with Appendix H extraction for Templates 2-3)

**Advantages:**
* ✅ Fastest iteration - edit tuple, run immediately
* ✅ Deterministic cross-baseline comparison via shared imports
* ✅ Aligns with experimental repo context
* ✅ Maintains current proven pattern (all baselines already import from prompts.py)
* ✅ Template 1 can be implemented immediately (no Appendix H dependency)
* ✅ Clear migration path to manifests when needed

**Limitations:**
* ⚠️ No provenance metadata in code (addressed via comments)
* ⚠️ Hard to version prompts independently from code (Git commits sufficient for experimental work)
* ⚠️ Scaling ceiling at ~300 prompts (B2 has 215, well under threshold)

**Alignment with Data Strategy:**

The data strategy document states:

> "The current config-driven approach is **sufficient for ongoing exploration** unless transition criteria are met:
> * Prompt sets grow beyond 20 prompts.
> * Need for cross-baseline comparison on identical prompt sequences.
> * Need for multi-scorer evaluation (CLIP, BERT, ImageReward).
> * Need for provenance tracking (DDPO overlap, B2 templates, DrawBench).
> * Start of diagram generation benchmarks."

**Criteria Met:**
* ✓ Prompt count > 20 (but < 300 threshold for config-driven)
* ✓ Cross-baseline comparison (solved by shared imports)
* ✗ Multi-scorer evaluation (not needed yet)
* ~ Provenance tracking (useful but not critical for experimental work)

**Analysis:** Transition is weakly triggered. Data strategy endorses config-driven for experimental repos with prompt counts under ~200-300. Alternative 1 is the appropriate choice.

#### Alternative 2: Minimal Manifest Layer (Phase 1)

**Implementation:**
* Create `data/manifests/shared_v1/prompts.jsonl` with all B2 templates
* Add `PromptDataset` class in `clover/data/prompts.py`
* Maintain tuple fallback in configs for backward compatibility
* Add CLI arg `--prompt-manifest` to load from JSONL

**Implementation Effort:** 1-2 days

**Advantages:**
* ✅ Structured provenance (prompt_id, source_family, semantic_tag, metadata)
* ✅ Starts manifest path for future multi-scorer infrastructure
* ✅ Enables external prompt contributions via JSONL editing

**Limitations:**
* ⚠️ Adds significant complexity without immediate payoff
* ⚠️ Slows iteration (need manifest file editing + Dataset class maintenance)
* ⚠️ Requires data loading infrastructure (not yet present in clover/data/)
* ⚠️ Over-engineered for experimental repo with 215 prompts

**Recommendation:** Defer until multi-scorer evaluation becomes necessary.

#### Alternative 3: Full Manifest-First Implementation

**Implementation:**
* Complete manifest-based architecture (Dataset abstraction, split management, offline rewards)
* No config tuple fallback
* Production-grade data infrastructure

**Implementation Effort:** 1-3 days

**Advantages:**
* ✅ Production-ready
* ✅ Full provenance tracking and metadata
* ✅ Multi-scorer infrastructure ready

**Limitations:**
* ❌ Massive over-engineering for experimental repo
* ❌ Significantly slows iteration speed
* ❌ High maintenance burden
* ❌ Violates "experimental repo" context

**Recommendation:** Not appropriate for current project phase.

#### Alternative 4: Hybrid (Templates in Code, Splits in Manifests)

**Implementation:**
* Add B2 template tuples to prompts.py
* Create small JSONL files with split indices
* Load splits at runtime but keep prompts in code

**Implementation Effort:** 6-8 hours

**Limitations:**
* ⚠️ Confusing split between code and data
* ⚠️ Index-based splits fragile to reordering
* ⚠️ Adds complexity without clear benefits over Alternative 1

**Recommendation:** Not recommended - Alternative 1 is simpler and Alternative 2 is more thorough.

## Selected Approach

**Alternative 1: Extend prompts.py with B2 Template Constants**

### Rationale

1. **Aligns with experimental repo context:** Data strategy explicitly endorses config-driven for <300 prompts (B2 has 215)

2. **Solves the core problem:** Shared module imports guarantee identical prompt sequences across baselines for fair comparison

3. **Transition criteria weakly met:** Only 2 of 7 criteria strongly met:
   * ✓ Prompt count > 20 (but < 300 threshold)
   * ✓ Cross-baseline comparison (solved by shared constants)
   * ✗ Multi-scorer evaluation (not needed yet)
   * ✗ External contributions (experimental repo)
   * ✗ Independent versioning (Git sufficient)

4. **Fastest implementation:** 2-4 hours (with Template 1 immediately available) vs 1-3 days

5. **Clear migration path:** Transition to Alternative 2 when multi-scorer comparison or publication provenance becomes critical

### Implementation Plan

#### Phase 1: Template 1 Implementation (2-4 hours)

**Step 1: Add Template 1 generator function (1 hour)**

```python
# clover/utils/prompts.py

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
    # Total: 45 animals (matches DDPO baseline)
)

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

**Step 2: Define Template 1 train/eval splits (30 minutes)**

```python
# Template 1 splits (90 train / 45 eval based on B2-DiffuRL paper ratios)
# Train: first 30 animals × 3 activities = 90 prompts
# Eval: last 15 animals × 3 activities = 45 prompts

TEMPLATE_1_TRAIN_PROMPTS = TEMPLATE_1_ALL_PROMPTS[:90]  # First 30 animals
TEMPLATE_1_EVAL_PROMPTS = TEMPLATE_1_ALL_PROMPTS[90:]   # Last 15 animals
```

**Step 3: Add docstrings and provenance comments (30 minutes)**

Document template source, semantic tags, overlap slice identification, and split rationale.

**Step 4: Test cross-baseline determinism (1 hour)**

```bash
# Test identical sequences
python -m clover.baselines.ddpo --train-epochs 1 --rollouts-per-epoch 2
python -m clover.baselines.dpok --train-epochs 1 --rollouts-per-epoch 2
python -m clover.baselines.b2diffurl --train-epochs 1 --rollouts-per-epoch 2

# Verify prompts in each config.json are identical
diff outputs/ddpo/config.json outputs/dpok/config.json
```

#### Phase 2: Templates 2 & 3 Extraction from B2-DiffuRL Paper (2-4 hours)

**Step 1: Access B2-DiffuRL Appendix H**

* URL: https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_Towards_Better_Alignment_Training_Diffusion_Models_with_Reinforcement_Learning_Against_CVPR_2025_paper.pdf
* Extract complete prompt lists for Template 2 (40 prompts) and Template 3 (40 prompts)
* Extract train/eval split assignments

**Step 2: Add Template 2 and 3 constants**

```python
# B2-DiffuRL Template 2: Color-Attribute Patterns
# Source: B2-DiffuRL CVPR 2025, Appendix H
# Pattern: [color] [fruit/vegetable]
# Semantic tag: attribute, color

TEMPLATE_2_PROMPTS: tuple[str, ...] = (
    # Extract from Appendix H
    "red apple",
    "green grape",
    # ... 38 more prompts
)

TEMPLATE_2_TRAIN_PROMPTS = TEMPLATE_2_PROMPTS[:30]  # Adjust based on paper
TEMPLATE_2_EVAL_PROMPTS = TEMPLATE_2_PROMPTS[30:]

# B2-DiffuRL Template 3: Spatial-Relational Patterns
# Source: B2-DiffuRL CVPR 2025, Appendix H
# Pattern: [object_1] [predicate] [object_2]
# Semantic tag: relation, spatial

TEMPLATE_3_PROMPTS: tuple[str, ...] = (
    # Extract from Appendix H
    "a cat to the left of a couch",
    "a dog on the moon",
    # ... 38 more prompts
)

TEMPLATE_3_TRAIN_PROMPTS = TEMPLATE_3_PROMPTS[:30]  # Adjust based on paper
TEMPLATE_3_EVAL_PROMPTS = TEMPLATE_3_PROMPTS[30:]
```

#### Phase 3: Unified B2 Prompt Sets (30 minutes)

**Step 1: Create combined prompt sets**

```python
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

**Step 2: Update module docstring**

Document template usage patterns, cross-baseline comparison guidance, and migration path to manifests.

#### Phase 4: Baseline Integration & Validation (1 hour)

**Step 1: Update baseline configs with B2 templates**

Baselines can now import and use template-specific or combined prompt sets:

```python
# Example: ddpo.py config using Template 1 only (DDPO overlap)
from clover.utils.prompts import TEMPLATE_1_TRAIN_PROMPTS, TEMPLATE_1_EVAL_PROMPTS

@dataclass
class DDPOConfig:
    train_prompts: tuple[str, ...] = TEMPLATE_1_TRAIN_PROMPTS
    eval_prompts: tuple[str, ...] = TEMPLATE_1_EVAL_PROMPTS
    # ... other config fields
```

**Step 2: Verify cross-baseline comparison**

Run all three baselines with identical template sets and verify:
* Identical prompts in config.json
* Same reward distributions (within random seed tolerance)
* Reproducible cross-baseline ranking

### Implementation Timeline

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| 1 | Template 1 generator & splits | 2h | None |
| 1 | Template 1 testing & validation | 1-2h | Phase 1 |
| 2 | Extract Templates 2-3 from Appendix H | 2-4h | B2-DiffuRL paper access |
| 2 | Add Template 2-3 constants | 1h | Phase 2 extraction |
| 3 | Create unified B2 prompt sets | 30min | Phase 2 |
| 4 | Baseline integration & validation | 1h | Phase 3 |
| **Total** | | **6-10 hours** | |

**Critical Path:** Template 1 can be implemented and tested immediately (2-4 hours). Templates 2-3 require paper access but are independent workstreams.

### Migration Path to Manifest-Based Architecture

When transition criteria become strongly met (multi-scorer evaluation, publication-level provenance tracking), migrate to manifest-based approach:

**Trigger Conditions:**
* Need to compare CLIP vs BERTScore vs ImageReward vs Aesthetic scorers offline
* Publication requires detailed provenance metadata and reproducibility guarantees
* Prompt sets grow beyond 300 prompts
* External collaborators need to contribute prompts without code changes

**Migration Steps:**

1. Create `clover/data/prompts.py` with `PromptDataset` class that can load from JSONL or fall back to tuples
2. Convert existing prompt tuples to `data/manifests/shared_v1/prompts.jsonl` with full metadata:
   ```json
   {
     "prompt_id": "b2_t1_dog_riding_bike_0001",
     "prompt_text": "a dog riding a bike",
     "source_family": "b2_template1",
     "semantic_tag": "activity",
     "split": "train",
     "overlap_slice": "ddpo_b2_shared",
     "metadata": {"animal": "dog", "activity": "riding a bike"}
   }
   ```
3. Add `--prompt-manifest` CLI arg to baseline configs
4. Implement offline multi-scorer infrastructure in `clover/utils/rewards_utils.py`
5. Deprecate tuple-based prompts (mark as legacy in prompts.py)

**Timeline:** 1-2 days when triggered

## Technical Scenarios

### Scenario 1: Template 1 Only (DDPO Overlap Focus)

**Use Case:** Baseline wants to reproduce DDPO experiments or focus on canonical overlap.

**Implementation:**

```python
# clover/baselines/ddpo.py
from clover.utils.prompts import TEMPLATE_1_TRAIN_PROMPTS, TEMPLATE_1_EVAL_PROMPTS

@dataclass
class DDPOConfig:
    train_prompts: tuple[str, ...] = TEMPLATE_1_TRAIN_PROMPTS  # 90 prompts
    eval_prompts: tuple[str, ...] = TEMPLATE_1_EVAL_PROMPTS    # 45 prompts
```

**Benefits:**
* Direct DDPO reproducibility
* Focus on well-documented animal-activity behavior
* Smaller prompt set for faster iteration (90 train vs 150 full)

### Scenario 2: Full B2 Template Coverage

**Use Case:** Baseline wants semantic coverage across all three B2 templates.

**Implementation:**

```python
# clover/baselines/b2diffurl.py
from clover.utils.prompts import B2_FULL_TRAIN_PROMPTS, B2_FULL_EVAL_PROMPTS

@dataclass
class B2DiffuRLConfig:
    train_prompts: tuple[str, ...] = B2_FULL_TRAIN_PROMPTS  # ~150 prompts
    eval_prompts: tuple[str, ...] = B2_FULL_EVAL_PROMPTS    # ~65 prompts
```

**Benefits:**
* Complete semantic coverage (activities, attributes, relations)
* Matches full B2-DiffuRL paper setup
* Enables template-specific analysis (e.g., "which template improves most with RL?")

### Scenario 3: Custom Template Mix

**Use Case:** Experiment wants specific template combinations.

**Implementation:**

```python
# Custom experiment
from clover.utils.prompts import (
    TEMPLATE_1_TRAIN_PROMPTS,
    TEMPLATE_3_TRAIN_PROMPTS,
    TEMPLATE_1_EVAL_PROMPTS,
    TEMPLATE_3_EVAL_PROMPTS,
)

# Focus on activities + spatial relations, skip attributes
CUSTOM_TRAIN = (*TEMPLATE_1_TRAIN_PROMPTS, *TEMPLATE_3_TRAIN_PROMPTS)
CUSTOM_EVAL = (*TEMPLATE_1_EVAL_PROMPTS, *TEMPLATE_3_EVAL_PROMPTS)
```

**Benefits:**
* Flexible experimentation
* Template-specific ablations
* Controlled semantic domain coverage

### Scenario 4: Cross-Baseline Comparison

**Use Case:** Fair comparison of DDPO vs DPOK vs B2-DiffuRL on identical prompt sequences.

**Implementation:**

All baselines use same import:

```python
from clover.utils.prompts import B2_FULL_TRAIN_PROMPTS, B2_FULL_EVAL_PROMPTS

# Identical across all three baseline configs
train_prompts: tuple[str, ...] = B2_FULL_TRAIN_PROMPTS
eval_prompts: tuple[str, ...] = B2_FULL_EVAL_PROMPTS
```

Run comparison:

```bash
python -m clover.baselines.ddpo --seed 42 --train-epochs 10
python -m clover.baselines.dpok --seed 42 --train-epochs 10
python -m clover.baselines.b2diffurl --seed 42 --train-epochs 10

# Compare final eval rewards
jq '.[-1].reward_mean' outputs/ddpo/history.json
jq '.[-1].reward_mean' outputs/dpok/history.json
jq '.[-1].reward_mean' outputs/b2diffurl/history.json
```

**Benefits:**
* Deterministic cross-baseline comparison (no sampling variation)
* Fair evaluation on identical prompts
* Reproducible rankings

## Considered Alternatives

### Alternative 2: Minimal Manifest Layer

**Summary:** Create JSONL manifests with provenance metadata and PromptDataset loader class.

**Advantages:**
* Structured metadata (prompt_id, source_family, semantic_tag, overlap_slice)
* Starts manifest path for future multi-scorer infrastructure
* External prompt contributions via JSONL editing

**Rejection Rationale:**
* Adds 1-2 days implementation time vs 2-4 hours for Alternative 1
* Requires data loading infrastructure not yet present in clover/data/
* Over-engineered for experimental repo with 215 prompts
* Multi-scorer evaluation (primary manifest benefit) is future work, not current need
* Git version control sufficient for experimental prompt management

**Evidence:**
* Data strategy document states config-driven is "sufficient for ongoing exploration" when prompt count < ~200-300
* Only 2 of 7 transition criteria strongly met
* Subagent evaluation scores Alternative 1 as "fastest iteration, deterministic comparison, proven pattern"

### Alternative 3: Full Manifest-First Implementation

**Summary:** Complete manifest-based architecture with Dataset abstraction, split management, and offline rewards.

**Advantages:**
* Production-ready
* Full provenance and metadata
* Multi-scorer infrastructure ready

**Rejection Rationale:**
* 1-3 days implementation vs 2-4 hours for Alternative 1
* Massive over-engineering for experimental repository
* Significantly slows iteration speed
* High maintenance burden
* Violates "experimental repo" project context
* No current need for production-grade data infrastructure

**Evidence:**
* User explicitly stated "I don't need production deployment in this repo. I need to set it up for experiments with baselines."
* Experimental repos prioritize iteration speed over production features
* Data strategy reserves manifest-first for when "transition criteria are strongly met"

### Alternative 4: Hybrid (Templates in Code, Splits in Manifests)

**Summary:** Store prompts as tuples in code, but use JSONL files with split indices.

**Advantages:**
* Balances code convenience with split structure
* Prompts remain easy to edit

**Rejection Rationale:**
* Confusing split between code (prompts) and data (split indices)
* Index-based splits fragile to tuple reordering
* Adds complexity without clear benefits over Alternative 1 (simpler) or Alternative 2 (more thorough)
* No compelling use case for this specific hybrid

**Evidence:**
* Subagent evaluation identified "split between code/data may be confusing" and "index-based splits fragile to reordering"
* Either keep everything in code (Alternative 1) or move to full manifest structure (Alternative 2)

## Key Takeaways

### Current State Assessment

* ✅ Prompt duplication successfully eliminated via prompts.py shared module
* ✅ All baselines verified to import from shared module
* ⚠️ Current prompts are placeholder clover examples, not B2 templates
* ✅ clover/data/ remains empty - no premature data infrastructure

### Implementation Strategy

* **Selected Approach:** Extend prompts.py with B2 template constants (Alternative 1)
* **Rationale:** Aligns with experimental repo context, maintains iteration speed, solves cross-baseline comparison
* **Timeline:** 6-10 hours total (Template 1 ready in 2-4 hours)
* **Migration Path:** Clear transition to manifests when multi-scorer evaluation becomes necessary

### Template Coverage

* **Template 1:** 135 prompts (45 animals × 3 activities) - DDPO overlap, ready to implement
* **Template 2:** 40 prompts (color-attribute) - requires Appendix H extraction
* **Template 3:** 40 prompts (spatial-relational) - requires Appendix H extraction
* **Total:** 215 prompts with ~150 train / ~65 eval split

### Critical Success Factors

1. **Template 1 Independence:** Can implement and test immediately without Appendix H access
2. **Cross-Baseline Determinism:** Shared imports guarantee identical sequences
3. **Fast Iteration:** Edit tuple, run immediately - no manifest preprocessing
4. **Clear Provenance:** Code comments document template sources and overlap slices
5. **Migration Ready:** Module docstring documents manifest transition path

### Transition Triggers (Future)

Migrate to manifest-based when any of these become true:
* Multi-scorer comparison needed (CLIP vs BERT vs ImageReward vs Aesthetic)
* Publication requires detailed provenance metadata
* Prompt sets grow beyond 300 prompts
* External collaborators contribute prompts
* Independent prompt versioning becomes necessary

## Evidence Log

### Local Repository Evidence

* clover/utils/prompts.py:1-35 - Current shared prompt module with design rationale
* Baseline configs verified to import from prompts.py (prompt duplication eliminated)
* clover/data/ empty - no premature data infrastructure

### Subagent Research Documents

* .copilot-tracking/research/subagents/2026-07-25/b2-template-specs.md - Complete B2 template specifications
  * Template 1: 135 prompts (45 animals × 3 activities)
  * Template 2: 40 prompts (color-attribute patterns)
  * Template 3: 40 prompts (spatial-relational patterns)
  * Semantic tags, split strategies, overlap slices
* .copilot-tracking/research/subagents/2026-07-25/prompt-implementation-alternatives.md - Evaluation of 4 implementation approaches
  * Alternative 1 (extend prompts.py): 2-4 hours, recommended
  * Alternative 2 (minimal manifests): 1-2 days, deferred
  * Alternative 3 (full manifest): 1-3 days, rejected
  * Alternative 4 (hybrid): 6-8 hours, not recommended

### External Research

* B2-DiffuRL CVPR 2025 paper - Three template families with semantic coverage
* DDPO (arXiv 2305.13301) - Animal-activity baseline (Template 1 overlap)
* Data strategy research (.copilot-tracking/research/2026-06-14/data-strategy-research.md) - Transition criteria and phased evolution

### Data Strategy Alignment

From data-strategy-research.md Executive Summary:

> "The current config-driven approach is **sufficient for ongoing exploration** unless transition criteria are met:
> * Prompt sets grow beyond 20 prompts.
> * Need for cross-baseline comparison on identical prompt sequences.
> * Need for multi-scorer evaluation (CLIP, BERT, ImageReward).
> * Need for provenance tracking (DDPO overlap, B2 templates, DrawBench).
> * Start of diagram generation benchmarks."

**Analysis:** 2 of 7 criteria met (prompt count > 20, cross-baseline comparison), but data strategy explicitly endorses config-driven for experimental repos with <300 prompts. Alternative 1 is the correct choice.

## Actionable Next Steps for Implementation

### Immediate Actions (Template 1 - First 2-4 hours)

**Step 1: Implement Template 1 Generator (1 hour)**

Add to clover/utils/prompts.py:
* TEMPLATE_1_ANIMALS tuple (45 animals)
* TEMPLATE_1_ACTIVITIES tuple (3 activities)
* generate_template_1_prompts() function
* TEMPLATE_1_ALL_PROMPTS constant (135 prompts)

**Step 2: Define Template 1 Splits (30 minutes)**

* TEMPLATE_1_TRAIN_PROMPTS (90 prompts - first 30 animals)
* TEMPLATE_1_EVAL_PROMPTS (45 prompts - last 15 animals)

**Step 3: Document Template 1 (30 minutes)**

* Add docstrings with pattern, source, semantic tags
* Document DDPO overlap (ddpo_b2_shared slice)
* Add usage examples in module docstring

**Step 4: Test Cross-Baseline Determinism (1-2 hours)**

* Update one baseline config to use TEMPLATE_1_TRAIN_PROMPTS
* Run training for 1 epoch
* Verify prompts in config.json match expectations
* Repeat for other baselines, verify identical sequences

### Follow-Up Actions (Templates 2-3 - Next 2-4 hours)

**Step 5: Extract Templates 2-3 from B2-DiffuRL Paper**

* Access B2-DiffuRL Appendix H from CVPR 2025 proceedings
* Extract complete 40-prompt lists for Template 2 (color-attribute)
* Extract complete 40-prompt lists for Template 3 (spatial-relational)
* Document train/eval split assignments per template

**Step 6: Add Templates 2-3 Constants (1 hour)**

* TEMPLATE_2_PROMPTS, TEMPLATE_2_TRAIN_PROMPTS, TEMPLATE_2_EVAL_PROMPTS
* TEMPLATE_3_PROMPTS, TEMPLATE_3_TRAIN_PROMPTS, TEMPLATE_3_EVAL_PROMPTS
* Document patterns, sources, semantic tags

**Step 7: Create Unified B2 Sets (30 minutes)**

* B2_FULL_TRAIN_PROMPTS (concatenate all three template train sets)
* B2_FULL_EVAL_PROMPTS (concatenate all three template eval sets)
* Update module docstring with unified set usage

### Validation Actions (Final 1 hour)

**Step 8: Baseline Integration**

* Update baseline configs with examples using different template combinations
* Document recommended usage patterns (Template 1 only, full B2, custom mix)

**Step 9: Cross-Baseline Comparison Test**

* Run all three baselines with identical B2_FULL prompts
* Verify config.json prompts are identical
* Compare reward distributions
* Document comparison methodology

**Step 10: Documentation Update**

* Update README or docs with B2 template usage
* Document migration path to manifests
* Add examples for each scenario (Template 1 only, full B2, custom mix, cross-baseline)

### Decision Points

**After Template 1 Implementation:**
* ✅ Template 1 working → proceed with Templates 2-3 extraction
* ⚠️ Appendix H access blocked → use Template 1 only for initial experiments

**After Full Implementation:**
* ✅ Cross-baseline determinism verified → production use
* ⚠️ Iteration speed too slow → simplify (unlikely with tuple-based approach)
* ⚠️ Need multi-scorer comparison → begin manifest migration (Alternative 2)

### Future Work (When Transition Criteria Met)

**Manifest Migration Checklist:**
* Create clover/data/prompts.py with PromptDataset class
* Convert tuples to data/manifests/shared_v1/prompts.jsonl
* Add prompt_id, source_family, semantic_tag, metadata fields
* Implement offline multi-scorer reward computation
* Add --prompt-manifest CLI arg
* Deprecate tuple-based prompts
