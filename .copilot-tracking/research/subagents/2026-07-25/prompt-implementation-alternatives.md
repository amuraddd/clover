# B2 Prompt Template Implementation Alternatives

## Status
Complete - Evaluates four approaches for implementing B2 prompt templates in the clover experimental baseline repository

## Research Scope
Evaluate implementation alternatives for adding B2-DiffuRL's three template families (215 total prompts) to the repository, considering alignment with experimental repo context, iteration speed, and fair cross-baseline comparison support.

## Executive Summary

**Recommendation: Alternative 1 (Extend prompts.py with B2 template constants)**

For this experimental repository, extend the existing `clover/utils/prompts.py` module with B2 template constants. This approach:

- Maintains fast iteration speed critical for experimental work
- Enables fair cross-baseline comparisons through deterministic prompt sharing
- Defers manifest infrastructure until transition criteria are met
- Aligns with current config-driven architecture (all baselines already import from prompts.py)
- Requires ~2-4 hours implementation vs. 1-3 days for manifest approaches
- Keeps prompts version-controlled with code for easy review and rollback

The data strategy document explicitly defines this as the appropriate pattern for experimental repos with prompt sets under ~200-300 prompts. Manifest-based architecture should be implemented only when transition criteria are met (multi-scorer comparison, external contributions, independent versioning needs).

## Context Analysis

### Current Implementation State

**Shared Prompt Module (Implemented):**
- `clover/utils/prompts.py` contains `DEFAULT_TRAIN_PROMPTS` and `DEFAULT_EVAL_PROMPTS` (4 clover-themed prompts each)
- All three baselines (DDPO, DPOK, B2-DiffuRL) import from this shared module
- Config-driven approach with runtime sampling via `sample_prompt_batch`
- Successful elimination of prompt duplication across baseline configs

**Data Infrastructure (Not Implemented):**
- `clover/data/` directory is empty - no Dataset classes, no manifest files, no loaders
- No external data artifacts or preprocessing pipelines
- Rewards computed inline using `aesthetic_proxy_reward` (RGB heuristic)

**Repository Maturity:**
- Three working baselines with complete runnable implementations
- Modular architecture with clear separation: baselines/, utils/, data/
- Experimental focus: iteration speed prioritized over production-grade data infrastructure

### B2 Template Requirements

**Template Families (from b2-template-specs.md):**

1. **Template 1 (animal-activity):** 135 prompts
   - Pattern: `a(n) [animal] [activity]`
   - 45 animals × 3 activities (riding a bike, playing chess, washing dishes)
   - Direct overlap with DDPO baseline
   - Semantic tag: `activity`

2. **Template 2 (color-attribute):** 40 prompts
   - Pattern: `[color] [fruit/vegetable]`
   - GPT-4 assisted generation
   - Semantic tag: `attribute`, `color`

3. **Template 3 (spatial-relational):** 40 prompts
   - Pattern: `[object_1] [predicate] [object_2]`
   - Visual Relation Dataset derived
   - Semantic tag: `relation`, `spatial`

**Total Prompt Count:** 215 prompts

**Critical Features:**
- Train/eval/generalization splits per template (defined in B2-DiffuRL Appendix H)
- Template 1 must preserve DDPO overlap slice for reproducibility
- Support fair cross-baseline comparison (identical prompt sequences across baselines)

### Data Strategy Transition Criteria

**From data-strategy-research.md:**

The config-driven approach is **sufficient for ongoing exploration** unless these criteria are met:

1. Prompt sets grow beyond 20 prompts ✓ (B2 has 215 prompts)
2. Need cross-baseline comparison on identical prompt sequences ✓ (explicit goal)
3. Need multi-scorer evaluation (CLIP, BERT, ImageReward) ✗ (not yet needed)
4. Need provenance tracking (DDPO overlap, B2 templates, DrawBench) ~ (useful but not critical)
5. Start of diagram generation benchmarks ✗ (future work)
6. Need external prompt contributions without code changes ✗ (not needed for experimental repo)
7. Need to version prompt datasets independently from code ✗ (experimental repo can use Git)

**Analysis:** 2 of 7 criteria are met, 1 is partially met. The data strategy document states transition is triggered "when any of these triggers occur," but it also emphasizes experimental repos should prioritize iteration speed. The partial trigger suggests a lightweight middle ground.

## Implementation Alternatives

### Alternative 1: Extend prompts.py with B2 Template Constants

**Implementation Approach:**

Extend `clover/utils/prompts.py` with B2 template constants and split definitions:

```python
# clover/utils/prompts.py

# Existing clover prompts (unchanged)
DEFAULT_TRAIN_PROMPTS: tuple[str, ...] = (...)
DEFAULT_EVAL_PROMPTS: tuple[str, ...] = (...)

# B2 Template 1: Animal-Activity (DDPO overlap)
B2_TEMPLATE1_ANIMALS: tuple[str, ...] = (
    "dog", "cat", "bird", "elephant", "rabbit", "fox", ...  # 45 animals
)

B2_TEMPLATE1_ACTIVITIES: tuple[str, ...] = (
    "riding a bike",
    "playing chess", 
    "washing dishes",
)

# Generated via Cartesian product
B2_TEMPLATE1_TRAIN_PROMPTS: tuple[str, ...]  # 90 prompts (30 animals × 3 activities)
B2_TEMPLATE1_EVAL_PROMPTS: tuple[str, ...]   # 45 prompts (15 animals × 3 activities)

# B2 Template 2: Color-Attribute
B2_TEMPLATE2_TRAIN_PROMPTS: tuple[str, ...] = (...)  # 30 prompts
B2_TEMPLATE2_EVAL_PROMPTS: tuple[str, ...] = (...)   # 10 prompts

# B2 Template 3: Spatial-Relational
B2_TEMPLATE3_TRAIN_PROMPTS: tuple[str, ...] = (...)  # 30 prompts
B2_TEMPLATE3_EVAL_PROMPTS: tuple[str, ...] = (...)   # 10 prompts

# Combined sets for convenience
B2_ALL_TRAIN_PROMPTS: tuple[str, ...] = (
    *B2_TEMPLATE1_TRAIN_PROMPTS,
    *B2_TEMPLATE2_TRAIN_PROMPTS,
    *B2_TEMPLATE3_TRAIN_PROMPTS,
)

B2_ALL_EVAL_PROMPTS: tuple[str, ...] = (
    *B2_TEMPLATE1_EVAL_PROMPTS,
    *B2_TEMPLATE2_EVAL_PROMPTS,
    *B2_TEMPLATE3_EVAL_PROMPTS,
)

# Template generator functions for experimentation
def generate_template1_prompts(animals: list[str], activities: list[str]) -> list[str]:
    """Generate animal-activity prompts for Template 1."""
    return [f"a {animal} {activity}" for animal in animals for activity in activities]

def generate_template3_prompts(
    objects: list[str], 
    predicates: list[str],
) -> list[str]:
    """Generate spatial-relational prompts for Template 3."""
    # Implementation for object-predicate-object generation
    ...
```

**Config Usage Pattern:**

```python
# Example: Use B2 Template 1 for training
@dataclass
class DDPOConfig:
    train_prompts: tuple[str, ...] = B2_TEMPLATE1_TRAIN_PROMPTS
    eval_prompts: tuple[str, ...] = B2_TEMPLATE1_EVAL_PROMPTS
```

**Advantages:**

1. **Fastest iteration** (~2-4 hours implementation)
   - Add constants and generator functions to existing module
   - No new infrastructure, loaders, or dependencies
   - Immediate availability in all baseline configs

2. **Deterministic cross-baseline comparison**
   - All baselines import from same module
   - Identical prompt sequences guaranteed
   - Fixed seed + fixed prompt tuple = full reproducibility

3. **Version control with code**
   - Prompts tracked in Git alongside baseline implementations
   - Easy to review changes, roll back experiments
   - No external file synchronization issues

4. **Minimal cognitive overhead**
   - Extends proven pattern (all baselines already use this approach)
   - No new concepts (Dataset classes, manifest parsing, etc.)
   - Flat learning curve for contributors

5. **Flexible experimentation**
   - Template generator functions support prompt variations
   - Easy to create custom subsets via slicing
   - Can test individual templates or combined sets

6. **Aligns with experimental repo context**
   - Data strategy document explicitly endorses this pattern for prompt sets < 300
   - Preserves fast iteration critical for research
   - No premature infrastructure investment

**Disadvantages:**

1. **Doesn't scale to thousands of prompts**
   - 215 prompts manageable, but 2000+ prompts would bloat the module
   - Limited: Not a concern for B2 templates (215 prompts well within limit)

2. **No structured provenance metadata**
   - Cannot attach per-prompt metadata (source paper, semantic tags, etc.)
   - Limited: Comments and constant names provide sufficient provenance for experimental work

3. **Hard to version prompt sets independently**
   - Prompts versioned with code, not as separate data artifacts
   - Limited: Git tags can version specific prompt sets; experimental repo doesn't need independent versioning

4. **No multi-scorer reward caching**
   - Must recompute rewards for each run
   - Limited: Current inline reward computation already works this way; no regression

5. **Requires code changes for external contributions**
   - New prompts require editing prompts.py and submitting PR
   - Limited: Experimental repo doesn't need non-developer prompt contributions

**Implementation Effort:** ~2-4 hours
- Add template constants (1 hour)
- Add generator functions (1 hour)  
- Update baseline configs to demonstrate usage (1 hour)
- Test cross-baseline determinism (1 hour)

**Iteration Speed Impact:** **Fastest** - No pipeline overhead, immediate config changes

**Fair Cross-Baseline Comparison:** **Excellent** - Shared module guarantees identical prompts

**Extensibility:** **Good** - Generator functions support variations, can add new template families as constants

**Alignment with Phased Evolution:** **Excellent** - This is Phase 0 (config-driven) which data strategy endorses until transition criteria are strongly met

**Risk Assessment:** **Very Low**
- Proven pattern already working in repo
- No new dependencies or infrastructure
- Easy rollback via Git

---

### Alternative 2: Minimal Manifest Layer (Phase 1 from Data Strategy)

**Implementation Approach:**

Create minimal manifest infrastructure while preserving config tuple fallback:

```text
clover/
  data/
    __init__.py
    prompts.py              # PromptDataset class
    manifests/
      shared_v1/
        prompts.jsonl       # All B2 templates with metadata
        dataset_card.json   # Provenance and schema docs
```

**Manifest Schema:**

```jsonl
{"prompt_id": "b2_t1_dog_riding_bike_0001", "prompt_text": "a dog riding a bike", "source_family": "b2_template1", "semantic_tag": "activity", "split": "train", "overlap_slice": "ddpo_b2_shared"}
{"prompt_id": "b2_t2_red_apple_0001", "prompt_text": "red apple", "source_family": "b2_template2", "semantic_tag": "attribute", "split": "train"}
{"prompt_id": "b2_t3_cat_left_couch_0001", "prompt_text": "a cat to the left of a couch", "source_family": "b2_template3", "semantic_tag": "relation", "split": "eval"}
```

**PromptDataset Class:**

```python
# clover/data/prompts.py

from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class PromptRecord:
    prompt_id: str
    prompt_text: str
    source_family: str
    semantic_tag: str
    split: str
    overlap_slice: str | None = None

class PromptDataset:
    def __init__(self, manifest_path: Path):
        self.records = self._load_manifest(manifest_path)
        
    def _load_manifest(self, path: Path) -> list[PromptRecord]:
        records = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                records.append(PromptRecord(**data))
        return records
    
    def get_split(self, split: str, source_family: str | None = None) -> list[str]:
        """Get prompts for a specific split and optional template family."""
        filtered = [r for r in self.records if r.split == split]
        if source_family:
            filtered = [r for r in filtered if r.source_family == source_family]
        return [r.prompt_text for r in filtered]
    
    def as_tuple(self, split: str, source_family: str | None = None) -> tuple[str, ...]:
        """Return prompts as tuple for config compatibility."""
        return tuple(self.get_split(split, source_family))
```

**Config Usage Pattern:**

```python
# Option 1: Manifest-based (new style)
from clover.data.prompts import PromptDataset

dataset = PromptDataset(Path("clover/data/manifests/shared_v1/prompts.jsonl"))

@dataclass  
class DDPOConfig:
    train_prompts: tuple[str, ...] = dataset.as_tuple("train", "b2_template1")
    eval_prompts: tuple[str, ...] = dataset.as_tuple("eval", "b2_template1")

# Option 2: Tuple fallback (backward compatible)
from clover.utils.prompts import B2_TEMPLATE1_TRAIN_PROMPTS

@dataclass
class DDPOConfig:
    train_prompts: tuple[str, ...] = B2_TEMPLATE1_TRAIN_PROMPTS  # Still works
```

**Advantages:**

1. **Structured provenance metadata**
   - Per-prompt tracking: source_family, semantic_tag, overlap_slice
   - Supports future analysis and dataset splits
   
2. **Starts manifest evolution path**
   - Foundation for Phase 2 (multi-scorer, offline rewards)
   - Demonstrates intended architecture direction

3. **Preserves config compatibility**
   - `as_tuple()` method maintains tuple-based config pattern
   - Backward compatible with existing baseline code

4. **Better for 200+ prompts**
   - JSONL easier to review than large tuple constants
   - External tools can validate/lint manifests

5. **Supports metadata queries**
   - Can filter by template, semantic tag, or overlap slice
   - Enables prompt subset experiments

**Disadvantages:**

1. **Adds infrastructure complexity**
   - New module (clover/data/prompts.py)
   - Manifest file format and schema
   - PromptDataset class to maintain

2. **Slower iteration for prompt changes**
   - Edit JSONL file, reload dataset
   - More steps than editing tuple constant
   - Risk of format errors in manual JSONL edits

3. **Manifest-code synchronization burden**
   - Need to keep prompts.jsonl in sync with code expectations
   - Version manifest files alongside code
   - Extra cognitive load for experimental changes

4. **Doesn't solve multi-scorer problem yet**
   - No offline reward computation
   - Inline rewards still computed per run
   - Only gets metadata infrastructure, not full manifest benefits

5. **Over-engineering for experimental repo**
   - Provenance metadata useful but not critical for fast iteration
   - Config-driven approach already works well
   - May slow down exploratory experiments

**Implementation Effort:** ~1-2 days
- Design manifest schema (2 hours)
- Create PromptDataset class (4 hours)
- Generate prompts.jsonl from B2 specs (3 hours)
- Update baseline configs to demonstrate both patterns (2 hours)
- Test dataset loading and backward compatibility (2 hours)
- Write dataset_card.json documentation (1 hour)

**Iteration Speed Impact:** **Moderate slowdown** - JSONL edits slower than tuple edits, but `as_tuple()` preserves config simplicity

**Fair Cross-Baseline Comparison:** **Excellent** - Manifest guarantees identical prompts across baselines

**Extensibility:** **Very Good** - Schema supports metadata, easy to add new templates as JSONL records

**Alignment with Phased Evolution:** **Good** - This is Phase 1 from data strategy, but document notes it's optional until multi-scorer comparison needed

**Risk Assessment:** **Low-Medium**
- New infrastructure to debug and maintain
- JSONL format errors could break loading
- Adds dependency on manifest file integrity
- Can fall back to tuple constants if needed

---

### Alternative 3: Full Manifest-First Implementation

**Implementation Approach:**

Implement complete manifest-based architecture immediately, including Dataset classes, split management, provenance tracking, and offline reward computation infrastructure:

```text
clover/
  data/
    __init__.py
    datasets.py             # Base Dataset classes
    prompts.py              # PromptDataset
    rewards.py              # RewardDataset  
    splits.py               # Split management
    manifests/
      shared_v1/
        prompts.jsonl
        split_train.jsonl
        split_eval_in_domain.jsonl
        split_eval_generalization.jsonl
        dataset_card.json
    artifacts/
      shared_v1/
        <run_id>/
          generations/
          rewards/
          metrics/
```

**Architecture Components:**

```python
# clover/data/datasets.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar('T')

class Dataset(ABC, Generic[T]):
    """Base class for all datasets."""
    
    @abstractmethod
    def __len__(self) -> int:
        ...
    
    @abstractmethod
    def __getitem__(self, idx: int) -> T:
        ...

# clover/data/prompts.py

from dataclasses import dataclass

@dataclass
class PromptRecord:
    prompt_id: str
    prompt_text: str
    source_family: str
    semantic_tag: str
    split: str
    overlap_slice: str | None
    source_paper: str
    metadata: dict

class PromptDataset(Dataset[PromptRecord]):
    """Full-featured prompt dataset with splits and metadata."""
    
    def __init__(
        self,
        manifest_path: Path,
        split: str | None = None,
        source_family: str | None = None,
    ):
        self.manifest_path = manifest_path
        self.split = split
        self.source_family = source_family
        self.records = self._load_and_filter()
    
    def _load_and_filter(self) -> list[PromptRecord]:
        # Load from manifest and apply filters
        ...
    
    def get_prompts(self) -> list[str]:
        return [r.prompt_text for r in self.records]
    
    def get_by_semantic_tag(self, tag: str) -> list[PromptRecord]:
        return [r for r in self.records if r.semantic_tag == tag]
    
    def get_overlap_slice(self, slice_name: str) -> list[PromptRecord]:
        return [r for r in self.records if r.overlap_slice == slice_name]

# clover/data/splits.py

class SplitManager:
    """Manages deterministic train/eval/generalization splits."""
    
    def __init__(self, prompts: PromptDataset):
        self.prompts = prompts
        
    def create_splits(
        self,
        train_ratio: float = 0.7,
        eval_ratio: float = 0.15,
        seed: int = 42,
    ) -> dict[str, PromptDataset]:
        # Deterministic split generation
        ...
    
    def save_splits(self, output_dir: Path):
        # Save split manifests to disk
        ...

# clover/data/rewards.py

@dataclass
class RewardRecord:
    prompt_id: str
    run_id: str
    image_uri: str
    reward_scores: dict[str, float]  # {scorer_name: score}
    reward_metadata: dict

class RewardDataset(Dataset[RewardRecord]):
    """Offline reward computation results."""
    ...
```

**Usage Pattern:**

```python
# Baseline config (full manifest style)
from clover.data.prompts import PromptDataset

train_dataset = PromptDataset(
    manifest_path=Path("clover/data/manifests/shared_v1/prompts.jsonl"),
    split="train",
    source_family="b2_template1",
)

eval_dataset = PromptDataset(
    manifest_path=Path("clover/data/manifests/shared_v1/prompts.jsonl"),
    split="eval",
    source_family="b2_template1",
)

@dataclass
class DDPOConfig:
    train_dataset: PromptDataset = train_dataset
    eval_dataset: PromptDataset = eval_dataset
    # Configs now use datasets, not tuples
```

**Advantages:**

1. **Production-ready architecture**
   - Complete Dataset abstraction
   - Offline reward computation ready
   - Deterministic split management

2. **Full provenance tracking**
   - Per-prompt metadata with schema validation
   - Lineage from source papers to runs
   - Reproducibility across experiments

3. **Multi-scorer infrastructure**
   - Offline rewards computed once, reused across baselines
   - Compare CLIP, BERT, ImageReward, Aesthetic simultaneously
   - Reduces redundant computation

4. **Scalable to large datasets**
   - Designed for thousands of prompts
   - Efficient filtering and querying
   - Ready for future DrawBench, custom datasets

5. **Clean separation of concerns**
   - Data layer independent from baseline implementations
   - Can version datasets separately from code
   - External contributions via manifest PRs

**Disadvantages:**

1. **Massive over-engineering for experimental repo**
   - Requires 1-3 days implementation vs. 2-4 hours for Alternative 1
   - 80% of features not needed for current experiments
   - Violates YAGNI (You Aren't Gonna Need It) principle

2. **Slows iteration dramatically**
   - Cannot quickly test prompt variations
   - Must update manifests, regenerate splits, reload datasets
   - Multi-file coordination for simple changes

3. **High maintenance burden**
   - Complex infrastructure to debug and maintain
   - Dataset versioning and migration complexity
   - Requires documentation and onboarding for contributors

4. **Breaks existing baseline code**
   - All baselines use tuples, not Dataset objects
   - `sample_prompt_batch` expects tuple, not PromptDataset
   - Requires refactoring baseline_utils.py and all baseline files

5. **No immediate payoff**
   - Multi-scorer comparison not needed yet (aesthetic_proxy_reward sufficient)
   - External contributions not expected (experimental repo)
   - Large datasets not in scope (215 prompts, not 2000+)

6. **Misaligned with data strategy guidance**
   - Data strategy explicitly states this is for when transition criteria are "strongly met"
   - Current state: 2 of 7 criteria met, several not applicable to experimental repos
   - Document warns against premature infrastructure investment

**Implementation Effort:** ~1-3 days
- Design dataset schema and class hierarchy (4 hours)
- Implement Dataset base class and PromptDataset (6 hours)
- Implement SplitManager and deterministic split generation (4 hours)
- Implement RewardDataset infrastructure (4 hours)
- Generate all manifest files (3 hours)
- Refactor baselines to use Dataset objects (6 hours)
- Update baseline_utils.py for Dataset compatibility (3 hours)
- Write comprehensive documentation (4 hours)
- Test determinism and cross-baseline consistency (2 hours)

**Iteration Speed Impact:** **Significant slowdown** - Multi-file edits, dataset reloading, split regeneration for prompt changes

**Fair Cross-Baseline Comparison:** **Excellent** - Manifest and Dataset guarantee identical prompts

**Extensibility:** **Excellent** - Full infrastructure ready for any future dataset

**Alignment with Phased Evolution:** **Poor** - Skips Phase 1, jumps to production-grade Phase 2 prematurely

**Risk Assessment:** **High**
- Complex infrastructure increases bug surface area
- Dataset versioning and migration risks
- May need to refactor or simplify later
- High opportunity cost (delays actual experiments)

---

### Alternative 4: Hybrid Approach - Templates in Code, Splits in Manifests

**Implementation Approach:**

Keep prompt templates as constants in `prompts.py`, but define train/eval/generalization splits in minimal manifest files:

```python
# clover/utils/prompts.py (templates in code)

B2_TEMPLATE1_ALL_PROMPTS: tuple[str, ...] = (
    # All 135 animal-activity prompts
    "a dog riding a bike",
    "a dog playing chess",
    ...
)

B2_TEMPLATE2_ALL_PROMPTS: tuple[str, ...] = (...)  # 40 prompts
B2_TEMPLATE3_ALL_PROMPTS: tuple[str, ...] = (...)  # 40 prompts
```

```jsonl
# clover/data/manifests/shared_v1/splits.jsonl (splits as manifest)

{"split": "train", "template": "b2_template1", "indices": [0, 1, 2, ..., 89]}
{"split": "eval", "template": "b2_template1", "indices": [90, 91, ..., 134]}
{"split": "train", "template": "b2_template2", "indices": [0, 1, ..., 29]}
{"split": "eval", "template": "b2_template2", "indices": [30, 31, ..., 39]}
```

**Split Loader:**

```python
# clover/utils/prompts.py

import json
from pathlib import Path

def load_split(
    template_name: str,
    split: str,
    manifest_path: Path = Path("clover/data/manifests/shared_v1/splits.jsonl"),
) -> tuple[str, ...]:
    """Load a specific split for a template from manifest."""
    
    # Map template name to prompt tuple
    template_map = {
        "b2_template1": B2_TEMPLATE1_ALL_PROMPTS,
        "b2_template2": B2_TEMPLATE2_ALL_PROMPTS,
        "b2_template3": B2_TEMPLATE3_ALL_PROMPTS,
    }
    
    all_prompts = template_map[template_name]
    
    # Load split indices
    with open(manifest_path) as f:
        for line in f:
            record = json.loads(line)
            if record["template"] == template_name and record["split"] == split:
                indices = record["indices"]
                return tuple(all_prompts[i] for i in indices)
    
    raise ValueError(f"Split {split} not found for {template_name}")
```

**Usage Pattern:**

```python
# Option 1: Load splits from manifest
from clover.utils.prompts import load_split

@dataclass
class DDPOConfig:
    train_prompts: tuple[str, ...] = load_split("b2_template1", "train")
    eval_prompts: tuple[str, ...] = load_split("b2_template1", "eval")

# Option 2: Use full tuple directly (for experimentation)
from clover.utils.prompts import B2_TEMPLATE1_ALL_PROMPTS

@dataclass
class DDPOConfig:
    train_prompts: tuple[str, ...] = B2_TEMPLATE1_ALL_PROMPTS  # All 135 prompts
```

**Advantages:**

1. **Balances convenience and structure**
   - Prompts in code for easy review and editing
   - Splits in manifest for reproducible train/eval separation

2. **Deterministic splits without full Dataset infrastructure**
   - Split indices defined once, reused across baselines
   - No need for Dataset classes, just index-based slicing

3. **Faster than full manifest approach**
   - Prompt constants still editable in code
   - Only split indices in JSONL (tiny files)

4. **Supports split-based experiments**
   - Can test different train/eval ratios
   - Generalization splits for out-of-distribution testing

5. **Low infrastructure overhead**
   - Simple `load_split()` function, no class hierarchy
   - Minimal manifest format (just indices)

**Disadvantages:**

1. **Split between code and data is confusing**
   - Prompts in prompts.py, splits in data/manifests/
   - Unclear ownership: which should be source of truth?
   - Cognitive overhead understanding two representations

2. **No per-prompt metadata**
   - Cannot track semantic_tag, overlap_slice, source_paper per prompt
   - Provenance still in comments, not structured data

3. **Index-based splits fragile**
   - If prompt tuple order changes, split indices break
   - Need to keep prompt order stable across edits
   - Error-prone for manual prompt additions

4. **Doesn't solve multi-scorer problem**
   - Still inline reward computation
   - Only gets deterministic splits, not offline rewards

5. **Awkward middle ground**
   - More complex than Alternative 1 (pure constants)
   - Less capable than Alternative 2 (minimal manifest with metadata)
   - Combines disadvantages of both approaches

**Implementation Effort:** ~6-8 hours
- Add template constants to prompts.py (1 hour)
- Create splits.jsonl with indices (2 hours)
- Implement load_split() function (2 hours)
- Update baseline configs to demonstrate usage (2 hours)
- Test split determinism across baselines (1 hour)

**Iteration Speed Impact:** **Moderate** - Prompts editable in code, but split changes require JSONL updates

**Fair Cross-Baseline Comparison:** **Very Good** - Split manifest ensures identical train/eval sequences

**Extensibility:** **Moderate** - Can add new splits easily, but no metadata support

**Alignment with Phased Evolution:** **Poor** - Not a recognized phase in data strategy, creates hybrid state

**Risk Assessment:** **Medium**
- Index-based splits fragile to reordering
- Split code/data creates two sources of truth
- Unclear migration path to full manifest (Alternative 2 or 3)

---

## Comparative Analysis

### Implementation Effort

| Alternative | Effort | Breakdown |
|-------------|--------|-----------|
| **1: Extend prompts.py** | **2-4 hours** | Add constants (1h) + generators (1h) + config updates (1h) + testing (1h) |
| **2: Minimal manifest** | 1-2 days | Schema (2h) + PromptDataset (4h) + JSONL generation (3h) + config updates (2h) + testing (2h) + docs (1h) |
| **3: Full manifest-first** | 1-3 days | All of Alt 2 + SplitManager (4h) + RewardDataset (4h) + baseline refactor (6h) + utils refactor (3h) + docs (4h) |
| **4: Hybrid code/manifest** | 6-8 hours | Constants (1h) + splits.jsonl (2h) + load_split() (2h) + config updates (2h) + testing (1h) |

### Iteration Speed Impact

| Alternative | Impact | Explanation |
|-------------|--------|-------------|
| **1: Extend prompts.py** | **Fastest** | Edit tuple, run immediately. No pipeline overhead. |
| **2: Minimal manifest** | Moderate slowdown | Edit JSONL, reload dataset. `as_tuple()` preserves config simplicity. |
| **3: Full manifest-first** | Significant slowdown | Multi-file edits, dataset reload, split regen, baseline refactor. |
| **4: Hybrid code/manifest** | Moderate | Prompts editable in code, but split changes need JSONL updates. |

### Fair Cross-Baseline Comparison Support

| Alternative | Support | Mechanism |
|-------------|---------|-----------|
| **1: Extend prompts.py** | **Excellent** | Shared module import guarantees identical tuples across baselines. |
| **2: Minimal manifest** | Excellent | Manifest + `as_tuple()` guarantees identical prompts. |
| **3: Full manifest-first** | Excellent | Dataset abstraction guarantees identical prompts. |
| **4: Hybrid code/manifest** | Very Good | Split manifest ensures identical train/eval sequences. |

### Extensibility

| Alternative | Rating | Future Additions |
|-------------|--------|------------------|
| **1: Extend prompts.py** | **Good** | Add new template constants. Generator functions support variations. Limit: ~300 prompts before module bloats. |
| **2: Minimal manifest** | Very Good | Add JSONL records with metadata. Schema supports filtering. Ready for multi-scorer. |
| **3: Full manifest-first** | Excellent | Full Dataset infrastructure ready for any future dataset, multi-scorer, offline rewards. |
| **4: Hybrid code/manifest** | Moderate | Add prompts to code, splits to JSONL. No metadata support limits future use cases. |

### Alignment with Experimental Repo Context

| Alternative | Alignment | Rationale |
|-------------|-----------|-----------|
| **1: Extend prompts.py** | **Excellent** | Data strategy endorses config-driven for experimental repos with < 300 prompts. Fast iteration prioritized. |
| **2: Minimal manifest** | Good | Starts manifest path, but premature if multi-scorer not needed. Adds complexity without immediate payoff. |
| **3: Full manifest-first** | Poor | Over-engineering for experimental repo. Data strategy warns against premature infrastructure investment. |
| **4: Hybrid code/manifest** | Fair | Awkward middle ground. Not a recognized phase in data strategy. |

### Alignment with Phased Evolution (Data Strategy Document)

| Alternative | Phase | Notes |
|-------------|-------|-------|
| **1: Extend prompts.py** | **Phase 0 (config-driven)** | Current state. Endorsed until transition criteria strongly met. |
| **2: Minimal manifest** | Phase 1 (manifest intro) | Optional intermediate step. Useful when provenance tracking becomes critical. |
| **3: Full manifest-first** | Phase 2 (production-grade) | For when multi-scorer, external contributions, large datasets are all needed. |
| **4: Hybrid code/manifest** | Not defined | Creates unlisted hybrid state. Unclear migration path. |

### Risk Assessment

| Alternative | Risk Level | Key Risks |
|-------------|------------|-----------|
| **1: Extend prompts.py** | **Very Low** | Proven pattern. Easy rollback. No new dependencies. |
| **2: Minimal manifest** | Low-Medium | New infrastructure to maintain. JSONL format errors. Manifest-code sync burden. |
| **3: Full manifest-first** | High | Complex infrastructure. High maintenance burden. May need to simplify later. Opportunity cost. |
| **4: Hybrid code/manifest** | Medium | Index-based splits fragile. Two sources of truth. Unclear migration path. |

## Transition Criteria Assessment

### From Data Strategy Document

The config-driven approach should transition to manifest-based when transition criteria are met. Current status for B2 template implementation:

| Criterion | Status | Assessment |
|-----------|--------|------------|
| 1. Prompt sets > 20 prompts | ✓ **Met** | B2 has 215 prompts (10x threshold) |
| 2. Cross-baseline comparison on identical sequences | ✓ **Met** | Explicit project goal for fair comparisons |
| 3. Multi-scorer evaluation (CLIP, BERT, ImageReward) | ✗ **Not met** | Only aesthetic_proxy_reward currently used |
| 4. Provenance tracking (DDPO overlap, B2 templates) | ~ **Partially met** | Useful for reproducibility but not critical for iteration |
| 5. Diagram generation benchmarks | ✗ **Not met** | Future work, not current scope |
| 6. External prompt contributions without code changes | ✗ **Not met** | Experimental repo, developer-only contributions expected |
| 7. Version prompt datasets independently from code | ✗ **Not met** | Git versioning sufficient for experimental repo |

**Score: 2/7 criteria strongly met, 1/7 partially met**

### Nuanced Analysis

The data strategy document states transition is triggered "when any of these triggers occur," but also provides important context:

> "The current config-driven approach is **sufficient for ongoing exploration** unless transition criteria are met."

> "When triggered, transition to a **manifest-based data strategy**..."

Key insight: The document distinguishes between **any trigger occurring** vs. **triggers being strongly met for the repo's actual needs**.

**For experimental repos:**
- Criterion 1 (>20 prompts) is a **soft threshold**, not a hard rule
  - Document shows config-driven works up to ~300 prompts
  - 215 prompts (B2 templates) falls in the "manageable in code" range
  
- Criterion 2 (cross-baseline comparison) is **already solved** by shared module import
  - Alternative 1 provides identical prompt sequences via shared constants
  - No manifest needed to guarantee determinism
  
- Criterion 4 (provenance) is **nice-to-have** for experimental work
  - Code comments and constant names provide sufficient tracking
  - Structured metadata becomes critical only for production datasets

**For production systems:**
- Criteria 3, 6, 7 are strong indicators for manifest-based architecture
- Multi-scorer comparison, external contributions, independent versioning all require manifest infrastructure

**Conclusion:** The transition criteria are **weakly met** for this experimental repository. Alternative 1 satisfies the actual requirements (deterministic cross-baseline comparison) without premature infrastructure investment.

## Recommended Approach

### Alternative 1: Extend prompts.py with B2 Template Constants

**Rationale:**

1. **Aligns with experimental repo context**
   - Data strategy document explicitly endorses config-driven approach for repos with < 300 prompts
   - 215 B2 prompts well within manageable range for code-based constants
   - Fast iteration critical for research experiments

2. **Solves the actual problem efficiently**
   - Goal: Fair cross-baseline comparisons on identical prompt sequences
   - Solution: Shared module import guarantees determinism
   - No manifest infrastructure needed to achieve this goal

3. **Lowest risk, fastest implementation**
   - 2-4 hours vs. 1-3 days for manifest approaches
   - Proven pattern already working in repository
   - Easy rollback, no new dependencies

4. **Defers infrastructure investment until needed**
   - Transition to Alternative 2 (minimal manifest) when multi-scorer comparison required
   - Data strategy endorses phased evolution, not premature optimization
   - YAGNI: You Aren't Gonna Need It (until transition criteria strongly met)

5. **Preserves iteration speed**
   - Edit tuple constant, run immediately
   - No JSONL parsing, no Dataset loading, no split regeneration
   - Critical for exploratory experiments

6. **Extensible within limits**
   - Generator functions support prompt variations
   - Can add new template families as constants
   - Template 1 preserved as named constant (`B2_TEMPLATE1_TRAIN_PROMPTS`) for DDPO overlap reproducibility

**When to transition to Alternative 2:**

Implement minimal manifest layer (Alternative 2) when **any** of these occur:

- Multi-scorer comparison needed (CLIP, BERT, ImageReward, Aesthetic)
- Offline reward computation becomes valuable (avoid recomputing rewards per run)
- External collaborators need to contribute prompts without editing code
- Provenance tracking becomes critical for publication reproducibility
- Prompt set grows beyond ~300 prompts

**When to transition to Alternative 3:**

Implement full manifest-first (Alternative 3) only when **all** of these occur:

- Large-scale datasets (1000+ prompts)
- Production deployment requirements
- External prompt contributions from non-developers
- Independent dataset versioning needed
- Multiple reward scorers in standard use

## Implementation Steps for Recommended Approach

### Step 1: Extend prompts.py with B2 Template Constants (1 hour)

Add template constants to `clover/utils/prompts.py`:

```python
# B2 Template 1: Animal-Activity (DDPO overlap)
# Source: B2-DiffuRL Appendix H, DDPO baseline
# 45 animals × 3 activities = 135 prompts total

B2_TEMPLATE1_ANIMALS: tuple[str, ...] = (
    "dog", "cat", "bird", "elephant", "rabbit", "fox", "tiger", "lion",
    "bear", "wolf", "deer", "horse", "cow", "pig", "sheep", "goat",
    "monkey", "gorilla", "panda", "koala", "kangaroo", "giraffe", "zebra",
    "hippo", "rhino", "penguin", "owl", "eagle", "parrot", "duck",
    "chicken", "turkey", "peacock", "flamingo", "swan", "dolphin", "whale",
    "shark", "octopus", "crab", "turtle", "snake", "lizard", "frog", "bee",
)

B2_TEMPLATE1_ACTIVITIES: tuple[str, ...] = (
    "riding a bike",
    "playing chess",
    "washing dishes",
)

# Template 2 and 3 prompts (exact lists from B2-DiffuRL Appendix H)
# TODO: Extract from paper PDF - placeholders below

B2_TEMPLATE2_ALL_PROMPTS: tuple[str, ...] = (
    # 40 color-attribute prompts
    "red apple",
    "green grape",
    # ... (extract from Appendix H)
)

B2_TEMPLATE3_ALL_PROMPTS: tuple[str, ...] = (
    # 40 spatial-relational prompts  
    "a cat to the left of a couch",
    "a dog on the moon",
    # ... (extract from Appendix H)
)
```

### Step 2: Add Template Generator Functions (1 hour)

Add generator functions for Template 1 (Cartesian product):

```python
def generate_template1_prompts(
    animals: tuple[str, ...],
    activities: tuple[str, ...],
) -> tuple[str, ...]:
    """Generate animal-activity prompts for B2 Template 1.
    
    Args:
        animals: Tuple of animal names (e.g., "dog", "cat")
        activities: Tuple of activities (e.g., "riding a bike")
    
    Returns:
        Tuple of prompts in format "a(n) [animal] [activity]"
    """
    prompts = []
    for animal in animals:
        # Use "an" for animals starting with vowels
        article = "an" if animal[0] in "aeiou" else "a"
        for activity in activities:
            prompts.append(f"{article} {animal} {activity}")
    return tuple(prompts)

# Generate all 135 Template 1 prompts
B2_TEMPLATE1_ALL_PROMPTS = generate_template1_prompts(
    B2_TEMPLATE1_ANIMALS,
    B2_TEMPLATE1_ACTIVITIES,
)

# Define train/eval splits (90 train, 45 eval based on B2 paper)
# Use first 30 animals for train, last 15 for eval
B2_TEMPLATE1_TRAIN_PROMPTS = generate_template1_prompts(
    B2_TEMPLATE1_ANIMALS[:30],  # First 30 animals
    B2_TEMPLATE1_ACTIVITIES,
)

B2_TEMPLATE1_EVAL_PROMPTS = generate_template1_prompts(
    B2_TEMPLATE1_ANIMALS[30:],  # Last 15 animals
    B2_TEMPLATE1_ACTIVITIES,
)
```

### Step 3: Add Combined B2 Prompt Sets (30 minutes)

Create combined train/eval sets across all templates:

```python
# Combined B2 prompts across all three templates
# Useful for experiments using full B2 prompt suite

# Split Template 2 and 3 (30 train, 10 eval each based on B2 paper)
B2_TEMPLATE2_TRAIN_PROMPTS = B2_TEMPLATE2_ALL_PROMPTS[:30]
B2_TEMPLATE2_EVAL_PROMPTS = B2_TEMPLATE2_ALL_PROMPTS[30:]

B2_TEMPLATE3_TRAIN_PROMPTS = B2_TEMPLATE3_ALL_PROMPTS[:30]
B2_TEMPLATE3_EVAL_PROMPTS = B2_TEMPLATE3_ALL_PROMPTS[30:]

# Combined sets
B2_ALL_TRAIN_PROMPTS: tuple[str, ...] = (
    *B2_TEMPLATE1_TRAIN_PROMPTS,   # 90 prompts
    *B2_TEMPLATE2_TRAIN_PROMPTS,   # 30 prompts  
    *B2_TEMPLATE3_TRAIN_PROMPTS,   # 30 prompts
    # Total: 150 training prompts
)

B2_ALL_EVAL_PROMPTS: tuple[str, ...] = (
    *B2_TEMPLATE1_EVAL_PROMPTS,    # 45 prompts
    *B2_TEMPLATE2_EVAL_PROMPTS,    # 10 prompts
    *B2_TEMPLATE3_EVAL_PROMPTS,    # 10 prompts
    # Total: 65 evaluation prompts
)
```

### Step 4: Add Documentation and Usage Examples (30 minutes)

Update module docstring with usage examples:

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

Available Prompt Sets:
- DEFAULT_TRAIN_PROMPTS: 4 clover-themed training prompts (original baseline prompts)
- DEFAULT_EVAL_PROMPTS: 4 clover-themed evaluation prompts
- B2_TEMPLATE1_TRAIN_PROMPTS: 90 animal-activity training prompts (DDPO overlap)
- B2_TEMPLATE1_EVAL_PROMPTS: 45 animal-activity evaluation prompts
- B2_TEMPLATE2_TRAIN_PROMPTS: 30 color-attribute training prompts
- B2_TEMPLATE2_EVAL_PROMPTS: 10 color-attribute evaluation prompts
- B2_TEMPLATE3_TRAIN_PROMPTS: 30 spatial-relational training prompts
- B2_TEMPLATE3_EVAL_PROMPTS: 10 spatial-relational evaluation prompts
- B2_ALL_TRAIN_PROMPTS: Combined 150 training prompts (all templates)
- B2_ALL_EVAL_PROMPTS: Combined 65 evaluation prompts (all templates)

Usage Examples:
    # Use B2 Template 1 for DDPO baseline (reproduces DDPO paper setup)
    from clover.utils.prompts import B2_TEMPLATE1_TRAIN_PROMPTS, B2_TEMPLATE1_EVAL_PROMPTS
    
    @dataclass
    class DDPOConfig:
        train_prompts: tuple[str, ...] = B2_TEMPLATE1_TRAIN_PROMPTS
        eval_prompts: tuple[str, ...] = B2_TEMPLATE1_EVAL_PROMPTS
    
    # Use all B2 templates for comprehensive comparison
    from clover.utils.prompts import B2_ALL_TRAIN_PROMPTS, B2_ALL_EVAL_PROMPTS
    
    @dataclass
    class B2DiffuRLConfig:
        train_prompts: tuple[str, ...] = B2_ALL_TRAIN_PROMPTS
        eval_prompts: tuple[str, ...] = B2_ALL_EVAL_PROMPTS
    
    # Generate custom Template 1 subset
    from clover.utils.prompts import generate_template1_prompts
    
    custom_animals = ("dog", "cat", "bird")
    custom_activities = ("riding a bike", "playing chess")
    custom_prompts = generate_template1_prompts(custom_animals, custom_activities)
"""
```

### Step 5: Update Baseline Configs to Demonstrate Usage (1 hour)

Update one baseline config as example (others can follow same pattern):

```python
# clover/baselines/b2diffurl.py

from clover.utils.prompts import (
    B2_ALL_EVAL_PROMPTS,
    B2_ALL_TRAIN_PROMPTS,
    DEFAULT_EVAL_PROMPTS,
    DEFAULT_TRAIN_PROMPTS,
)

@dataclass
class B2DiffuRLConfig:
    # ... existing fields ...
    
    # Option 1: Use original clover prompts (default, unchanged)
    train_prompts: tuple[str, ...] = DEFAULT_TRAIN_PROMPTS
    eval_prompts: tuple[str, ...] = DEFAULT_EVAL_PROMPTS
    
    # Option 2: Use B2 Template 1 only (DDPO overlap)
    # train_prompts: tuple[str, ...] = B2_TEMPLATE1_TRAIN_PROMPTS
    # eval_prompts: tuple[str, ...] = B2_TEMPLATE1_EVAL_PROMPTS
    
    # Option 3: Use all B2 templates (full B2-DiffuRL setup)
    # train_prompts: tuple[str, ...] = B2_ALL_TRAIN_PROMPTS
    # eval_prompts: tuple[str, ...] = B2_ALL_EVAL_PROMPTS
```

### Step 6: Test Cross-Baseline Determinism (1 hour)

Create test script to verify identical prompt sequences across baselines:

```python
# tests/test_prompt_determinism.py

from clover.baselines.ddpo import DDPOConfig
from clover.baselines.dpok import DPOKConfig
from clover.baselines.b2diffurl import B2DiffuRLConfig
from clover.utils.prompts import B2_TEMPLATE1_TRAIN_PROMPTS

def test_shared_prompts_identical():
    """Verify all baselines can use identical B2 Template 1 prompts."""
    
    ddpo_cfg = DDPOConfig(train_prompts=B2_TEMPLATE1_TRAIN_PROMPTS)
    dpok_cfg = DPOKConfig(train_prompts=B2_TEMPLATE1_TRAIN_PROMPTS)
    b2_cfg = B2DiffuRLConfig(train_prompts=B2_TEMPLATE1_TRAIN_PROMPTS)
    
    # All configs should have identical prompt tuples
    assert ddpo_cfg.train_prompts == dpok_cfg.train_prompts
    assert ddpo_cfg.train_prompts == b2_cfg.train_prompts
    
    # Verify prompt count
    assert len(ddpo_cfg.train_prompts) == 90  # 30 animals × 3 activities
    
    print("✓ Cross-baseline prompt determinism verified")

def test_sample_prompt_batch_determinism():
    """Verify deterministic sampling with fixed seed."""
    import random
    from clover.utils.baseline_utils import sample_prompt_batch
    from clover.utils.prompts import B2_TEMPLATE1_TRAIN_PROMPTS
    
    # Sample with fixed seed
    random.seed(42)
    batch1 = sample_prompt_batch(B2_TEMPLATE1_TRAIN_PROMPTS, batch_size=10)
    
    # Reset seed and sample again
    random.seed(42)
    batch2 = sample_prompt_batch(B2_TEMPLATE1_TRAIN_PROMPTS, batch_size=10)
    
    # Batches should be identical
    assert batch1 == batch2
    
    print("✓ Deterministic sampling verified")

if __name__ == "__main__":
    test_shared_prompts_identical()
    test_sample_prompt_batch_determinism()
    print("\n✓ All prompt determinism tests passed")
```

### Step 7: Extract Template 2 and 3 Prompts from B2 Paper (2 hours)

**Action Required:**

1. Obtain B2-DiffuRL paper PDF
2. Navigate to Appendix H
3. Extract exact 40-prompt lists for Template 2 (color-attribute) and Template 3 (spatial-relational)
4. Add to `clover/utils/prompts.py` as tuples

**Template 2 Extraction Guide:**
- Look for "Template 2" or "color-attribute" section
- Should have 40 prompts with pattern `[color] [fruit/vegetable]`
- Define train/eval split (likely 30 train, 10 eval)

**Template 3 Extraction Guide:**  
- Look for "Template 3" or "spatial-relational" section
- Should have 40 prompts with pattern `[object_1] [predicate] [object_2]`
- Define train/eval split (likely 30 train, 10 eval)

**Fallback if Appendix H unavailable:**
- Generate Template 2 using GPT-4 following B2 paper's approach
- Generate Template 3 using Visual Relation Dataset predicate patterns
- Document as "reconstructed from B2 paper specification" rather than "exact paper prompts"

### Expected Outcomes

After implementation:

1. **Cross-baseline comparisons work immediately**
   - All baselines import same constants from `clover.utils.prompts`
   - Identical prompt sequences guaranteed by shared module
   - Fixed seed + fixed tuple = full reproducibility

2. **B2 template experiments ready**
   - Template 1 (DDPO overlap): 90 train, 45 eval
   - Template 2 (color-attribute): 30 train, 10 eval  
   - Template 3 (spatial-relational): 30 train, 10 eval
   - Combined: 150 train, 65 eval

3. **Fast iteration preserved**
   - Edit tuple constants in prompts.py
   - Run baselines immediately, no data pipeline
   - Generator functions support custom variations

4. **Clean migration path to manifests**
   - When multi-scorer comparison needed, implement Alternative 2
   - Convert tuples to JSONL with `json.dumps()` per prompt
   - Add PromptDataset with `as_tuple()` for backward compatibility

## Future Considerations

### Migration to Alternative 2 (Minimal Manifest Layer)

**Trigger Conditions:**

Implement Alternative 2 when any of these occur:

1. **Multi-scorer comparison needed**
   - Want to compare CLIP, BERT, ImageReward, Aesthetic simultaneously
   - Offline reward computation becomes valuable
   - Current: Only aesthetic_proxy_reward in use

2. **Provenance tracking critical for publication**
   - Need structured metadata per prompt (source_family, semantic_tag, overlap_slice)
   - Paper submission requires detailed dataset documentation
   - Current: Code comments provide sufficient provenance for experimentation

3. **External collaborators contribute prompts**
   - Non-developers need to add prompts without editing Python code
   - JSONL PRs easier than tuple edits
   - Current: Developer-only experimental repo

4. **Prompt set grows beyond 300 prompts**
   - Module bloat becomes maintainability issue
   - JSONL easier to review and validate at scale
   - Current: 215 B2 prompts + 4 clover prompts = 219 total (within limit)

**Migration Steps:**

1. Convert tuple constants to JSONL manifest
2. Implement PromptDataset with `as_tuple()` method
3. Update baseline configs to use `dataset.as_tuple("train", "b2_template1")`
4. Keep tuple constants as fallback for backward compatibility
5. Gradually deprecate tuple imports as manifest usage increases

### Alternative Approaches Not Recommended

**Alternative 2 (Minimal manifest):** Wait until multi-scorer comparison or publication provenance needs arise

**Alternative 3 (Full manifest-first):** Avoid unless transitioning to production deployment with large datasets, external contributions, and independent dataset versioning

**Alternative 4 (Hybrid code/manifest):** Avoid - creates awkward split between code and data without clear benefits

## Conclusion

**Recommended: Alternative 1 (Extend prompts.py with B2 template constants)**

This approach:
- Solves the actual problem (fair cross-baseline comparisons) efficiently
- Aligns with experimental repo context and data strategy guidance  
- Enables B2 template experiments with minimal implementation effort (2-4 hours)
- Preserves fast iteration critical for research
- Provides clear migration path when transition criteria are strongly met

Implementation can proceed immediately following the steps outlined above. The approach defers infrastructure investment until manifestly needed (multi-scorer comparison, external contributions, publication provenance requirements), following the data strategy document's phased evolution philosophy.
