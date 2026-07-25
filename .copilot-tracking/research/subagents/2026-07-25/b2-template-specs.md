# B2-DiffuRL Prompt Template Specifications

## Status
Complete - Extracted from data strategy research and baseline assumptions research

## Research Scope
Extract and document the B2-DiffuRL prompt template specifications from the data strategy research document and related baseline research to support manifest-based dataset implementation.

## Executive Summary

B2-DiffuRL (CVPR 2025) defines three distinct prompt templates for training diffusion models with reinforcement learning:

- **Template 1 (animal-activity)**: Direct overlap with DDPO, 45 animals × 3 activities = 135 prompts
- **Template 2 (color-attribute)**: GPT-4 assisted generation, 40 prompts featuring color-fruit/vegetable combinations
- **Template 3 (spatial-relational)**: Visual Relation Dataset derived, 40 prompts with object-predicate-object structure

The templates provide semantic coverage across three key alignment domains: **activities**, **attributes**, and **spatial relations**.

## Template Specifications

### Template 1: Animal-Activity Patterns (DDPO Overlap)

**Pattern Structure:**
```
a(n) [animal] [activity]
```

**Template Variables:**
- `animal`: One of 45 common animals
- `activity`: One of 3 activities

**Activities (n=3):**
1. riding a bike
2. playing chess
3. washing dishes

**Animals (n=45):**
The exact list from DDPO baseline research:
- Common animals used uniformly across DDPO and B2-DiffuRL
- Same animal set used in DDPO aesthetic quality and prompt-image alignment experiments
- Sampled from 398 ImageNet-1000 animal categories narrowed to 45 most common

**Total Prompts:** 45 animals × 3 activities = **135 prompts**

**Semantic Tag:** `activity`

**Source Family:** `b2_template1`

**Overlap Slice:** `ddpo_b2_shared` - This is the canonical overlap between DDPO and B2-DiffuRL

**Example Prompts:**
```
a dog riding a bike
a cat playing chess
a bird washing dishes
an elephant riding a bike
a rabbit playing chess
a fox washing dishes
```

**Coverage Goals:**
- Test prompt-image alignment for action understanding
- Evaluate compositional generation (animal + activity)
- Known to encourage cartoon/style shortcuts (documented in DDPO and B2-DiffuRL papers)
- Strong for studying style collapse and prompt-image alignment behavior

**Paper Reference:**
- DDPO Section 5.2, 5.3, 6.1 - prompt-image alignment experiments
- B2-DiffuRL Section 4.1, Appendix H - Template 1 specification
- DDPO Appendix C, D, F - animal family definitions

**Notes:**
- This template represents the cleanest direct intersection with DDPO
- Should be preserved as a named overlap slice in any manifest implementation
- Training and held-out generalization splits defined in B2-DiffuRL Appendix H
- Narrow semantic domain but well-documented baseline behavior

### Template 2: Color-Attribute Patterns

**Pattern Structure:**
```
[color] [fruit/vegetable]
```

**Template Variables:**
- `color`: Color adjectives (exact list requires Appendix H)
- `fruit/vegetable`: Common fruits or vegetables

**Total Prompts:** **40 prompts**

**Generation Method:** GPT-4 assisted construction

**Semantic Tag:** `attribute`, `color`

**Source Family:** `b2_template2`

**Example Prompts (inferred from pattern):**
```
red apple
green grape
yellow banana
purple eggplant
orange carrot
blue blueberry
```

**Coverage Goals:**
- Test attribute binding (color to object)
- Evaluate basic compositional understanding
- Broaden beyond activity-based prompts to visual attributes
- Complement Template 1's action focus with static attribute alignment

**Paper Reference:**
- B2-DiffuRL Section 4.1 - Template 2 overview
- B2-DiffuRL Appendix H - full prompt lists (training and generalization)

**Notes:**
- Curated differently from Template 1 (GPT-4 assistance vs. direct animal list)
- Complements Template 1 by adding attribute-focused alignment tasks
- May have different difficulty profile for prompt-image alignment

**Gaps in Specification:**
- Exact color list not detailed in available research documents
- Exact fruit/vegetable list not detailed in available research documents
- Full prompt list requires reading B2-DiffuRL Appendix H from paper PDF

### Template 3: Spatial-Relational Patterns

**Pattern Structure:**
```
[object_1] [predicate] [object_2]
```

**Template Variables:**
- `object_1`: First object (subject)
- `predicate`: Spatial or relational predicate
- `object_2`: Second object (reference)

**Total Prompts:** **40 prompts**

**Source Dataset:** Visual Relation Dataset annotations

**Semantic Tag:** `relation`, `spatial`

**Source Family:** `b2_template3`

**Example Prompts (from research docs):**
```
a cat to the left of a couch
a dog on the moon
a book next to a lamp
```

**Predicates (inferred from Visual Relation Dataset):**
Common spatial relations include:
- `to the left of`
- `to the right of`
- `on`
- `under`
- `next to`
- `behind`
- `in front of`
- `above`
- `below`

**Coverage Goals:**
- Test spatial reasoning and relational understanding
- Evaluate compositional generation with explicit spatial constraints
- Assess multi-object scene composition
- Provide most challenging alignment task (relations harder than attributes or activities)

**Paper Reference:**
- B2-DiffuRL Section 4.1 - Template 3 overview
- B2-DiffuRL Appendix H - full prompt lists
- Visual Relation Dataset - source annotations for relation types

**Notes:**
- Based on Visual Relation Dataset annotations (different provenance from Templates 1-2)
- Likely most challenging template for prompt-image alignment
- Spatial predicates may require stronger compositional understanding than Template 1-2

**Gaps in Specification:**
- Exact object list not detailed in available research documents
- Exact predicate list not detailed in available research documents
- Full prompt list requires reading B2-DiffuRL Appendix H from paper PDF

## Semantic Coverage Matrix

| Template | Semantic Tags | Domain Focus | Prompt Count | Source |
|----------|--------------|--------------|--------------|--------|
| Template 1 | `activity` | Animal-activity composition | 135 (45×3) | DDPO overlap |
| Template 2 | `attribute`, `color` | Color-object binding | 40 | GPT-4 assisted |
| Template 3 | `relation`, `spatial` | Spatial relationships | 40 | Visual Relation Dataset |
| **Total** | | **Comprehensive alignment** | **215** | Multiple sources |

## Reward Function Specifications

B2-DiffuRL uses prompt-image alignment scores as rewards:

### BERTScore Route
```
generated image 
  → LLaVA v1.5 (image captioning)
  → DeBERTa/BERTScore (caption vs. original prompt)
  → scalar reward
```

### CLIPScore Route
```
generated image + original prompt
  → CLIP ViT-H/14 (text-image similarity)
  → scalar reward
```

**Comparison:**
- B2-DiffuRL paper includes analysis of reward stability between BERTScore and CLIPScore
- Synthetic dataset of 768 similar generated-image pairs used for metric comparison
- Both routes measure prompt-image alignment but with different architectures

## Split Definitions

**Training Prompts:**
- Used during RL fine-tuning
- Templates 1, 2, 3 each have designated training subsets

**Held-Out Generalization Prompts:**
- Separate from training prompts
- Used to evaluate generalization capability
- Defined per template in Appendix H

**Split Strategy:**
- Deterministic splits defined by paper (not random sampling)
- Same splits should be preserved for reproducibility
- `split` field values: `train`, `eval_in_domain`, `eval_generalization`

## Canonical Manifest Schema

Recommended JSONL record format for B2-DiffuRL prompts:

```json
{
  "prompt_id": "b2_t1_dog_riding_bike_0001",
  "prompt_text": "a dog riding a bike",
  "source_family": "b2_template1",
  "semantic_tag": "activity",
  "split": "train",
  "overlap_slice": "ddpo_b2_shared",
  "source_paper": "b2diffurl_cvpr2025",
  "license_note": "paper prompt list",
  "metadata": {
    "animal": "dog",
    "activity": "riding a bike"
  }
}
```

```json
{
  "prompt_id": "b2_t2_red_apple_0001",
  "prompt_text": "red apple",
  "source_family": "b2_template2",
  "semantic_tag": "attribute",
  "split": "train",
  "overlap_slice": null,
  "source_paper": "b2diffurl_cvpr2025",
  "license_note": "paper prompt list",
  "metadata": {
    "color": "red",
    "object": "apple"
  }
}
```

```json
{
  "prompt_id": "b2_t3_cat_left_of_couch_0003",
  "prompt_text": "a cat to the left of a couch",
  "source_family": "b2_template3",
  "semantic_tag": "relation",
  "split": "eval_generalization",
  "overlap_slice": null,
  "source_paper": "b2diffurl_cvpr2025",
  "license_note": "paper prompt list",
  "metadata": {
    "object_1": "cat",
    "predicate": "left of",
    "object_2": "couch"
  }
}
```

## Implementation Recommendations

### Phase 1: Template 1 Priority
1. Start with Template 1 as the default first benchmark slice
2. Reason: Cleanest overlap with DDPO, most documented baseline behavior
3. Use `ddpo_b2_shared` as the overlap slice identifier
4. Validate reward computation (BERTScore and CLIPScore) on this subset first

### Phase 2: Multi-Template Coverage
1. Add Templates 2 and 3 for broader semantic coverage
2. Maintain separate `source_family` fields for provenance tracking
3. Enable cross-template generalization analysis
4. Support multi-scorer comparison across all templates

### Phase 3: External Integration
1. Consider adding DPOK DrawBench prompts as `dpok_drawbench_extension` family
2. Keep B2 templates as core canonical subset
3. Enable template-specific reward analysis

## Prompt Diversity and Coverage Goals

**Activity Coverage (Template 1):**
- 3 distinct activities across 45 animals
- Tests compositional generation and activity understanding
- Known behavior: encourages cartoon/style shortcuts
- Coverage dimension: animal diversity × activity types

**Attribute Coverage (Template 2):**
- Color-object binding across 40 prompts
- Tests basic attribute alignment
- Coverage dimension: color variety × object types

**Relational Coverage (Template 3):**
- Spatial predicates across object pairs
- Tests multi-object composition and spatial reasoning
- Likely most challenging for alignment
- Coverage dimension: object pairs × spatial relations

**Combined Coverage:**
- 215 total prompts across 3 semantic domains
- Balanced mix of activities (63%), attributes (19%), relations (19%)
- Supports both in-domain and generalization evaluation

## Known Gaps and Clarifications Needed

### Gaps in Current Specification

1. **Template 1 Animal List:**
   - Source: 45 common animals from DDPO
   - Gap: Exact list not enumerated in research documents
   - Resolution: Extract from DDPO Appendix C/D or B2-DiffuRL Appendix H

2. **Template 2 Color List:**
   - Source: GPT-4 assisted generation
   - Gap: Exact color adjectives not specified
   - Resolution: Extract from B2-DiffuRL Appendix H

3. **Template 2 Fruit/Vegetable List:**
   - Source: GPT-4 assisted generation
   - Gap: Exact object list not specified
   - Resolution: Extract from B2-DiffuRL Appendix H

4. **Template 3 Object Lists:**
   - Source: Visual Relation Dataset annotations
   - Gap: Exact object_1 and object_2 lists not specified
   - Resolution: Extract from B2-DiffuRL Appendix H or Visual Relation Dataset

5. **Template 3 Predicate List:**
   - Source: Visual Relation Dataset annotations
   - Gap: Exact spatial predicates not specified
   - Resolution: Extract from B2-DiffuRL Appendix H or Visual Relation Dataset

6. **Split Definitions:**
   - Source: B2-DiffuRL Appendix H
   - Gap: Exact train/eval splits not detailed
   - Resolution: Extract from B2-DiffuRL Appendix H

7. **Prompt Counts Per Split:**
   - Source: Paper methodology
   - Gap: Not specified per template per split
   - Resolution: Extract from B2-DiffuRL Appendix H

### Required Next Actions

1. **Access B2-DiffuRL Paper PDF:**
   - URL: https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_Towards_Better_Alignment_Training_Diffusion_Models_with_Reinforcement_Learning_Against_CVPR_2025_paper.pdf
   - Target: Appendix H (prompt lists and split definitions)
   - Extract: Complete prompt lists for all three templates
   - Extract: Train/eval split assignments

2. **Validate Against DDPO Paper:**
   - URL: https://arxiv.org/pdf/2305.13301
   - Target: Appendices C, D, F (animal family definitions)
   - Validate: 45 animal list matches B2-DiffuRL Template 1

3. **Optional: Visual Relation Dataset:**
   - Consult Visual Relation Dataset for canonical predicate taxonomy
   - Validate Template 3 predicate choices
   - Understand annotation methodology

### Specification Completeness

**Currently Documented (High Confidence):**
- Template pattern structures ✓
- Semantic tags and categories ✓
- Total prompt counts per template ✓
- Source provenance per template ✓
- DDPO overlap identification ✓
- Reward function architectures ✓
- Manifest schema structure ✓

**Requires Paper Access (Medium Confidence):**
- Complete prompt lists for all templates
- Exact train/eval/generalization splits
- Prompt count distribution per split
- Template-specific generation parameters

**Inferred from Context (Lower Confidence):**
- Example prompts (based on pattern structure)
- Likely spatial predicates (based on Visual Relation Dataset conventions)
- Coverage goals (based on paper methodology)

## Related Documentation

### Source Documents
- Data Strategy Research: `.copilot-tracking/research/2026-06-14/data-strategy-research.md`
- Baseline Data Assumptions: `.copilot-tracking/research/subagents/2026-06-14/baseline-data-assumptions-research.md`

### External References
- B2-DiffuRL Paper: [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_Towards_Better_Alignment_Training_Diffusion_Models_with_Reinforcement_Learning_Against_CVPR_2025_paper.pdf)
- B2-DiffuRL arXiv: [arXiv:2503.11240v2](https://arxiv.org/html/2503.11240v2)
- DDPO Paper: [arXiv:2305.13301v4](https://arxiv.org/html/2305.13301v4)
- DDPO Project: [https://rl-diffusion.github.io/](https://rl-diffusion.github.io/)
- Visual Relation Dataset: Source for Template 3 relational annotations

### Implementation Context
- Current implementation uses config-driven prompts (4 clover-themed prompts)
- Baselines: `clover/baselines/ddpo.py`, `clover/baselines/dpok.py`, `clover/baselines/b2diffurl.py`
- Shared utilities: `clover/baselines/common.py`, `clover/utils/baseline_utils.py`
- Reward function: `clover/utils/rewards_utils.py` (currently aesthetic_proxy_reward)
- Data directory: `clover/data/` (currently empty, ready for manifest implementation)

## Usage Guidance

### For Manifest Implementation
1. Use this specification as the schema reference for `clover_shared_rl_subset_v1` manifest
2. Prioritize Template 1 extraction for DDPO/B2 overlap validation
3. Preserve `source_family` and `overlap_slice` fields for provenance
4. Implement deterministic splits matching paper methodology

### For Baseline Experiments
1. Start experiments with Template 1 (`ddpo_b2_shared` slice)
2. Compare BERTScore and CLIPScore reward functions
3. Validate against DDPO baseline results
4. Expand to Templates 2-3 for broader semantic coverage

### For Dataset Creation
1. Extract complete prompt lists from B2-DiffuRL Appendix H
2. Normalize into canonical JSONL manifest format
3. Encode metadata fields for template variables (animal, activity, color, etc.)
4. Create deterministic split manifests (train, eval_in_domain, eval_generalization)
5. Store in `data/manifests/clover_shared_rl_subset_v1/`

## Version History
- 2026-07-25: Initial specification extraction from research documents
- Research basis: Data strategy research (2026-06-14) and baseline assumptions research (2026-06-14)
