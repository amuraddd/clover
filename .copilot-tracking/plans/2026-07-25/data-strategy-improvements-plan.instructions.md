---
applyTo: '.copilot-tracking/changes/2026-07-25/data-strategy-improvements-changes.md'
---
<!-- markdownlint-disable-file -->
# Implementation Plan: Data Strategy Improvements

## Overview

Enhance the current config-driven data strategy with prompt deduplication, explicit evaluation splits, and configurable reward functions while maintaining the simple, self-contained architecture proven effective for baseline bring-up.

## Objectives

### User Requirements

* Improve maintainability of the current config-driven prompt strategy — Source: data-strategy-research.md Executive Summary and Actionable Next Steps
* Prepare the codebase for future evolution to manifest-based architecture when transition criteria are met — Source: data-strategy-research.md Evolution Path

### Derived Objectives

* Eliminate prompt duplication across baseline configs to reduce maintenance burden and prevent divergence — Derived from: research evidence showing identical train_prompts tuples in ddpo.py:51-56, dpok.py:51-56, b2diffurl.py:51-56
* Establish clear train/eval split boundaries to enable better evaluation methodology — Derived from: research limitation "No train/eval splits: Evaluation uses deduplicated train_prompts, not a held-out set"
* Enable reward function experimentation without code changes to each baseline — Derived from: research limitation "Single reward scorer: aesthetic_proxy_reward hardcoded; comparing CLIP vs ImageReward requires code changes"

## Context Summary

### Project Files

* clover/baselines/ddpo.py - DDPO implementation with hardcoded train_prompts tuple (lines 51-56)
* clover/baselines/dpok.py - DPOK implementation with duplicate train_prompts tuple (lines 51-56)
* clover/baselines/b2diffurl.py - B2-DiffuRL implementation with duplicate train_prompts tuple (lines 51-56)
* clover/baselines/common.py - Shared baseline utilities including make_reward_fn (lines 22-53)
* clover/utils/baseline_utils.py - Contains sample_prompt_batch (lines 289-300) and standard_eval_prompts (lines 640-653)
* clover/utils/rewards_utils.py - Contains aesthetic_proxy_reward implementation (lines 10-24)

### References

* .copilot-tracking/research/2026-06-14/data-strategy-research.md - Comprehensive data strategy research document with current state analysis and evolution roadmap

### Standards References

* vscode-local:/Users/makhan/.vscode/extensions/ise-hve-essentials.hve-core-3.2.2/.github/instructions/hve-core/markdown.instructions.md — Markdown formatting and YAML frontmatter requirements
* vscode-local:/Users/makhan/.vscode/extensions/ise-hve-essentials.hve-core-3.2.2/.github/instructions/hve-core/writing-style.instructions.md — Writing style and tone conventions

## Implementation Checklist

### [x] Implementation Phase 1: Create Shared Prompt Registry

<!-- parallelizable: true -->

* [x] Step 1.1: Create clover/utils/prompts.py with shared prompt definitions
  * Details: .copilot-tracking/details/2026-07-25/data-strategy-improvements-details.md (Lines 12-49)
* [x] Step 1.2: Add eval_prompts separate from train_prompts
  * Details: .copilot-tracking/details/2026-07-25/data-strategy-improvements-details.md (Lines 51-88)
* [x] Step 1.3: Validate phase changes
  * Run lint for clover/utils/prompts.py
  * Verify imports resolve correctly

### [x] Implementation Phase 2: Update Baseline Configs

<!-- parallelizable: false -->

* [x] Step 2.1: Update DDPO config to import from shared prompts
  * Details: .copilot-tracking/details/2026-07-25/data-strategy-improvements-details.md (Lines 102-140)
* [x] Step 2.2: Update DPOK config to import from shared prompts
  * Details: .copilot-tracking/details/2026-07-25/data-strategy-improvements-details.md (Lines 142-180)
* [x] Step 2.3: Update B2-DiffuRL config to import from shared prompts
  * Details: .copilot-tracking/details/2026-07-25/data-strategy-improvements-details.md (Lines 182-220)
* [x] Step 2.4: Validate phase changes
  * Run lint for all modified baseline files
  * Verify configs instantiate correctly with new imports

### [x] Implementation Phase 3: Add Configurable Reward Functions

<!-- parallelizable: true -->

* [x] Step 3.1: Extend rewards_utils.py with reward function registry
  * Details: .copilot-tracking/details/2026-07-25/data-strategy-improvements-details.md (Lines 234-280)
* [x] Step 3.2: Update make_reward_fn to support reward_type config parameter
  * Details: .copilot-tracking/details/2026-07-25/data-strategy-improvements-details.md (Lines 282-324)
* [x] Step 3.3: Add reward_type field to baseline configs
  * Details: .copilot-tracking/details/2026-07-25/data-strategy-improvements-details.md (Lines 326-360)
* [x] Step 3.4: Validate phase changes
  * Run lint for modified files
  * Verify reward function selection works correctly

### [x] Implementation Phase 4: Update Evaluation to Use Eval Prompts

<!-- parallelizable: false -->

* [x] Step 4.1: Update standard_eval_prompts to use config.eval_prompts when available
  * Details: .copilot-tracking/details/2026-07-25/data-strategy-improvements-details.md (Lines 381-432)
* [x] Step 4.2: Update baseline evaluate calls to use new eval logic
  * Details: .copilot-tracking/details/2026-07-25/data-strategy-improvements-details.md (Lines 434-461)
* [x] Step 4.3: Validate phase changes
  * Run lint for modified files
  * Verify evaluation uses correct prompt split

### [x] Implementation Phase 5: Validation

<!-- parallelizable: false -->

* [x] Step 5.1: Run full project validation
  * Execute all lint commands
  * Execute build/type checking
  * Run smoke test for each baseline with new configs
* [x] Step 5.2: Fix minor validation issues
  * Iterate on lint errors and type warnings
  * Apply fixes directly when corrections are straightforward
* [x] Step 5.3: Report blocking issues
  * Document any issues requiring additional research
  * Provide user with next steps
  * Avoid large-scale fixes within this phase

## Planning Log

See .copilot-tracking/plans/logs/2026-07-25/data-strategy-improvements-log.md for discrepancy tracking, implementation paths considered, and suggested follow-on work.

## Dependencies

* Python 3.9+
* Existing baseline implementations (ddpo.py, dpok.py, b2diffurl.py)
* Existing utility modules (baseline_utils.py, rewards_utils.py, common.py)

## Success Criteria

* All three baselines import prompts from shared registry, eliminating duplication — Traces to: research finding "Prompt duplication across baselines"
* Each baseline config defines separate train_prompts and eval_prompts tuples — Traces to: research recommendation "Add eval split"
* Reward function is configurable via reward_type config field without code changes — Traces to: research recommendation "Reward abstraction"
* All baselines maintain backward compatibility with existing experiments — Traces to: derived objective "maintain simple, self-contained architecture"
* Full project validation passes with no regressions — Traces to: quality standard "based on verified project conventions"
