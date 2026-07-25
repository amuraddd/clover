---
applyTo: '.copilot-tracking/changes/2026-07-25/b2-templates-changes.md'
---
<!-- markdownlint-disable-file -->
# Implementation Plan: B2 Prompt Templates

## Overview

Implement B2-DiffuRL's three prompt template families (215 total prompts) in clover/utils/prompts.py to enable fair cross-baseline comparisons with deterministic prompt sequences and semantic coverage across activities, attributes, and spatial relations.

## Objectives

### User Requirements

* Add B2-DiffuRL Template 1, 2, and 3 prompt families to the repository — Source: User request "implement B2-DiffuRL's three prompt template families"
* Enable fair cross-baseline comparison with identical prompt sequences — Source: Research requirement for deterministic cross-baseline evaluation
* Preserve DDPO overlap slice for reproducibility — Source: Research emphasis on Template 1 as canonical DDPO/B2-DiffuRL overlap

### Derived Objectives

* Maintain fast iteration speed appropriate for experimental work — Derived from: Repository context as experimental codebase, not production
* Define train/eval/generalization splits per template — Derived from: B2-DiffuRL paper methodology requiring proper evaluation splits
* Align implementation with phased data strategy evolution — Derived from: Data strategy research recommending config-driven for <300 prompts

## Context Summary

### Project Files

* clover/utils/prompts.py - Shared prompt module currently containing placeholder clover-themed prompts (4 train + 4 eval)
* clover/baselines/ddpo.py - DDPO baseline importing from shared prompts module
* clover/baselines/dpok.py - DPOK baseline importing from shared prompts module
* clover/baselines/b2diffurl.py - B2-DiffuRL baseline importing from shared prompts module

### References

* .copilot-tracking/research/2026-07-25/b2-templates-implementation-research.md - Complete research document with template specifications, implementation alternatives evaluation, and selected approach (Alternative 1: Extend prompts.py)
* .copilot-tracking/research/subagents/2026-07-25/b2-template-specs.md - Detailed B2 template specifications extracted from CVPR 2025 paper
* .copilot-tracking/research/subagents/2026-07-25/prompt-implementation-alternatives.md - Evaluation of 4 implementation approaches with rationale for selecting config-driven extension

### Standards References

* Data strategy research endorses config-driven approach for experimental repos with <300 prompts

## Implementation Checklist

### [x] Implementation Phase 1: Template 1 Generator and Splits

<!-- parallelizable: true -->

* [x] Step 1.1: Add Template 1 constants and generator function
  * Details: .copilot-tracking/details/2026-07-25/b2-templates-details.md (Lines 15-60)
* [x] Step 1.2: Define Template 1 train/eval splits
  * Details: .copilot-tracking/details/2026-07-25/b2-templates-details.md (Lines 62-85)
* [x] Step 1.3: Add docstrings and provenance comments
  * Details: .copilot-tracking/details/2026-07-25/b2-templates-details.md (Lines 87-105)
* [x] Step 1.4: Validate phase changes
  * Run Python syntax check on prompts.py
  * Skip baseline testing until Phase 3

### [x] Implementation Phase 2: Templates 2 and 3 Constants

<!-- parallelizable: false -->

* [x] Step 2.1: Extract Template 2 and 3 prompts from B2-DiffuRL Appendix H
  * Details: .copilot-tracking/details/2026-07-25/b2-templates-details.md (Lines 107-130)
  * Extracted from arXiv paper https://arxiv.org/pdf/2503.11240 Appendix H Tables 8 & 9
* [x] Step 2.2: Add Template 2 constants and splits
  * Details: .copilot-tracking/details/2026-07-25/b2-templates-details.md (Lines 132-155)
* [x] Step 2.3: Add Template 3 constants and splits
  * Details: .copilot-tracking/details/2026-07-25/b2-templates-details.md (Lines 157-180)
* [x] Step 2.4: Validate phase changes
  * Run Python syntax check on prompts.py
  * Verify total prompt counts match specifications (135 + 40 + 40 = 215)

### [x] Implementation Phase 3: Unified B2 Prompt Sets

<!-- parallelizable: true -->

* [x] Step 3.1: Create combined B2 train and eval prompt sets
  * Details: .copilot-tracking/details/2026-07-25/b2-templates-details.md (Lines 182-210)
* [x] Step 3.2: Update module docstring with B2 template usage
  * Details: .copilot-tracking/details/2026-07-25/b2-templates-details.md (Lines 212-235)
* [x] Step 3.3: Validate phase changes
  * Run linting on prompts.py
  * Verify combined sets have correct total counts

### [x] Implementation Phase 4: Baseline Integration and Testing

<!-- parallelizable: false -->

* [x] Step 4.1: Test Template 1 with one baseline
  * Details: .copilot-tracking/details/2026-07-25/b2-templates-details.md (Lines 237-260)
* [x] Step 4.2: Verify cross-baseline determinism
  * Details: .copilot-tracking/details/2026-07-25/b2-templates-details.md (Lines 262-285)
* [x] Step 4.3: Document usage patterns and examples
  * Details: .copilot-tracking/details/2026-07-25/b2-templates-details.md (Lines 287-310)

### [x] Implementation Phase 5: Validation

<!-- parallelizable: false -->

* [x] Step 5.1: Run full project validation
  * Execute Python linting on clover/utils/prompts.py
  * Verify import statements in all baseline configs
  * Check total prompt counts (215 total, ~150 train, ~65 eval)
* [x] Step 5.2: Fix minor validation issues
  * Iterate on lint errors and type hints
  * Apply fixes directly when corrections are straightforward
* [x] Step 5.3: Report blocking issues
  * Document issues requiring additional research
  * Provide user with next steps and recommended planning
  * Avoid large-scale fixes within this phase

## Planning Log

See .copilot-tracking/plans/logs/2026-07-25/b2-templates-log.md for discrepancy tracking, implementation paths considered, and suggested follow-on work.

## Dependencies

* Python 3.8+ (existing project requirement)
* Access to B2-DiffuRL CVPR 2025 paper Appendix H for Templates 2 and 3 prompt extraction
* Existing clover/utils/prompts.py module structure

## Success Criteria

* All three B2 template families implemented with train/eval splits — Traces to: User requirement for template implementation
* Template 1 preserves DDPO overlap for reproducibility — Traces to: Research emphasis on canonical overlap
* Cross-baseline determinism verified through shared imports — Traces to: User requirement for fair comparison
* Implementation maintains fast iteration speed — Traces to: Experimental repo context
* Total prompt count = 215 (135 + 40 + 40) with ~150 train / ~65 eval split — Traces to: B2-DiffuRL paper specifications
* Migration path to manifest-based architecture documented — Traces to: Data strategy phased evolution
