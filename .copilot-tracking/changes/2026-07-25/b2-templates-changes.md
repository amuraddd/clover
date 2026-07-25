<!-- markdownlint-disable-file -->
# Release Changes: B2 Prompt Templates

**Related Plan**: b2-templates-plan.instructions.md
**Implementation Date**: 2026-07-25

## Summary

Implementing B2-DiffuRL's three prompt template families (215 total prompts) in clover/utils/prompts.py to enable fair cross-baseline comparisons with deterministic prompt sequences.

## Changes

### Added

### Modified

* clover/utils/prompts.py - Added Template 1 section with B2-DiffuRL animal-activity patterns
  * Added TEMPLATE_1_ANIMALS tuple (45 animals)
  * Added TEMPLATE_1_ACTIVITIES tuple (3 activities: riding a bike, playing chess, washing dishes)
  * Added generate_template_1_prompts() function with correct article usage ("a" vs "an")
  * Added TEMPLATE_1_ALL_PROMPTS (135 prompts = 45 animals × 3 activities)
  * Added TEMPLATE_1_TRAIN_PROMPTS (90 prompts - first 30 animals)
  * Added TEMPLATE_1_EVAL_PROMPTS (45 prompts - last 15 animals)
  * Added comprehensive docstrings documenting pattern, source, semantic coverage, and split strategy
* clover/utils/prompts.py - Added Template 2 section with B2-DiffuRL color-attribute patterns
  * Added TEMPLATE_2_PROMPTS tuple (40 prompts: red apple, green apple, yellow banana, ... red grapefruit)
  * Added TEMPLATE_2_TRAIN_PROMPTS (30 prompts - first 30)
  * Added TEMPLATE_2_EVAL_PROMPTS (10 prompts - last 10)
  * Added comprehensive docstring documenting GPT-4 assisted generation, pattern [color] [fruit/vegetable], semantic coverage (attribute, color)
* clover/utils/prompts.py - Added Template 3 section with B2-DiffuRL spatial-relational patterns
  * Added TEMPLATE_3_PROMPTS tuple (40 prompts: chair under umbrella, ... person on the left of person)
  * Added TEMPLATE_3_TRAIN_PROMPTS (30 prompts - first 30)
  * Added TEMPLATE_3_EVAL_PROMPTS (10 prompts - last 10)
  * Added comprehensive docstring documenting Visual Relation Dataset source, pattern [object_1] [predicate] [object_2], semantic coverage (relation, spatial)
* clover/utils/prompts.py - Added unified B2 prompt sets combining all three templates
  * Added B2_FULL_TRAIN_PROMPTS (150 prompts: 90 Template 1 + 30 Template 2 + 30 Template 3)
  * Added B2_FULL_EVAL_PROMPTS (65 prompts: 45 Template 1 + 10 Template 2 + 10 Template 3)
  * Added comprehensive docstring documenting total counts (215), semantic coverage across all families
* clover/utils/prompts.py - Updated module-level docstring
  * Added B2-DiffuRL Templates section documenting all three templates with patterns and counts
  * Added cross-baseline comparison guidance for when to use individual vs unified sets
  * Added migration path documentation for future manifest-based architecture
  * Added usage examples section with three patterns: Template 1 only, full B2 coverage, custom template mixing

### Removed

## Additional or Deviating Changes

* Phase 2 (Templates 2 & 3) blocker resolved - prompts extracted from arXiv paper
  * Initial blocker: Cannot access B2-DiffuRL CVPR 2025 paper Appendix H via automated tools
  * Attempted: Direct PDF fetch, arXiv search, GitHub repository search (rate limited), HuggingFace search (auth errors)
  * Resolution: User provided arXiv link https://arxiv.org/pdf/2503.11240
  * Successfully extracted all 40 Template 2 prompts and 40 Template 3 prompts from Appendix H (Tables 8 & 9)
  * Status: Ready to resume Phase 2 implementation

## Release Summary

**Status**: Complete (5 of 5 phases complete)

**Completed Work:**
* ✅ Phase 1: Template 1 (Animal-Activity Patterns) - 135 prompts fully implemented with 90 train / 45 eval split
  * TEMPLATE_1_ANIMALS (45 elements)
  * TEMPLATE_1_ACTIVITIES (3 elements)
  * generate_template_1_prompts() function with correct article usage
  * TEMPLATE_1_ALL_PROMPTS, TEMPLATE_1_TRAIN_PROMPTS, TEMPLATE_1_EVAL_PROMPTS
  * Comprehensive docstrings and provenance comments
  * All validation passed (syntax, imports, counts, article usage, split integrity)

* ✅ Phase 2: Templates 2 & 3 - 80 prompts fully implemented with 30/10 split each
  * Template 2: 40 color-attribute prompts (red apple, yellow banana, etc.)
  * Template 3: 40 spatial-relational prompts (chair under umbrella, dog on the right of vase, etc.)
  * Both templates with comprehensive docstrings documenting source and semantic coverage
  * All validation passed

* ✅ Phase 3: Unified B2 Prompt Sets - Combined all templates for full semantic coverage
  * B2_FULL_TRAIN_PROMPTS (150 prompts: 90+30+30)
  * B2_FULL_EVAL_PROMPTS (65 prompts: 45+10+10)
  * Updated module docstring with B2-DiffuRL template documentation and usage examples
  * All validation passed

* ✅ Phase 4: Baseline Integration and Testing - Verified cross-baseline determinism
  * Template 1 validation: counts, splits, no overlap ✓
  * Cross-baseline determinism: memory address identity verified ✓
  * Usage examples added to module docstring ✓

* ✅ Phase 5: Final Validation - All systems verified
  * Python syntax check: passed ✓
  * Import verification: all templates import successfully ✓
  * Total counts verified: 215 prompts (150 train, 65 eval) ✓
  * Template breakdown: 135+40+40 ✓

**Files Affected:**
* Modified: 1 file ([clover/utils/prompts.py](clover/utils/prompts.py))
* Added: 0 files
* Removed: 0 files

**Implementation Statistics:**
* Total prompts: 215 (135 + 40 + 40)
* Train prompts: 150 (90 + 30 + 30)
* Eval prompts: 65 (45 + 10 + 10)
* Split ratio: 69.8% train / 30.2% eval
* Templates implemented: 3 (activity, attribute, spatial-relational)
* Semantic coverage: Complete B2-DiffuRL specification

**Template Details:**
* **Template 1 (activity, compositional):**
  * Pattern: "a(n) [animal] [activity]"
  * Total: 135 prompts (90 train, 45 eval)
  * Split: Animal-level separation (first 30 animals train, last 15 eval)
  * DDPO overlap: Preserved ddpo_b2_shared canonical slice

* **Template 2 (attribute, color):**
  * Pattern: "[color] [fruit/vegetable]"
  * Total: 40 prompts (30 train, 10 eval)
  * Source: GPT-4 assisted generation per B2-DiffuRL methodology

* **Template 3 (relation, spatial):**
  * Pattern: "[object_1] [predicate] [object_2]"
  * Total: 40 prompts (30 train, 10 eval)
  * Source: Visual Relation Dataset annotations

**Next Steps:**
Implementation is complete and ready for baseline use.

