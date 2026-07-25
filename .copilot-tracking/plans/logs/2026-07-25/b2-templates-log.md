<!-- markdownlint-disable-file -->
# Planning Log: B2 Prompt Templates

## Discrepancy Log

Gaps and differences identified between research findings and the implementation plan.

### Unaddressed Research Items

* DR-01: Full manifest-based implementation with provenance metadata
  * Source: .copilot-tracking/research/2026-07-25/b2-templates-implementation-research.md (Lines 420-450)
  * Reason: Data strategy explicitly endorses config-driven for experimental repos with <300 prompts. Transition criteria weakly met (only 2 of 7). Manifest infrastructure deferred until multi-scorer evaluation becomes necessary.
  * Impact: Low - config-driven approach sufficient for current experimental work, clear migration path documented

* DR-02: Offline multi-scorer reward computation infrastructure
  * Source: .copilot-tracking/research/subagents/2026-07-25/prompt-implementation-alternatives.md (Lines 85-95)
  * Reason: Multi-scorer evaluation (CLIP vs BERT vs ImageReward) is future work, not immediate need. Would require data loading infrastructure not yet present.
  * Impact: Low - single scorer sufficient for current baselines, noted as transition trigger

### Plan Deviations from Research

* DD-01: Templates 2 and 3 require manual extraction from Appendix H
  * Research recommends: Extraction as separate step
  * Plan implements: Extraction documented as Phase 2, Step 2.1 with explicit URL and instructions
  * Rationale: Template 1 can be implemented immediately without Appendix H dependency. Templates 2-3 are independent workstream that can proceed in parallel or sequentially.

* DD-02: Phase 2 blocked - automated PDF extraction failed (RESOLVED)
  * Plan specifies: Automated extraction from https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_Towards_Better_Alignment_Training_Diffusion_Models_with_Reinforcement_Learning_Against_CVPR_2025_paper.pdf
  * Actual blocker: PDF parsing tools unable to extract text from CVF Open Access repository
  * Impact: Phases 2-5 blocked pending manual extraction or alternative source
  * Attempted: Direct PDF fetch, arXiv search, GitHub search (rate limited), HuggingFace search (auth error)
  * Resolution: User provided arXiv link (https://arxiv.org/pdf/2503.11240), successfully extracted from HTML version
  * Status: RESOLVED - Templates 2 & 3 prompts extracted from Appendix H (Tables 8 & 9, pages 24-25)
  * Added: 2026-07-25 during Phase 2 execution, Resolved: 2026-07-25

## Implementation Paths Considered

### Selected: Config-Driven Extension of prompts.py (Alternative 1)

* Approach: Extend existing clover/utils/prompts.py with B2 template constants, generator functions, and split definitions
* Rationale: 
  * Aligns with experimental repo context (iteration speed critical)
  * Data strategy explicitly endorses config-driven for <300 prompts (B2 has 215)
  * Solves cross-baseline comparison through deterministic shared imports
  * Fastest implementation (6-10 hours vs 1-3 days for manifests)
  * Clear migration path when multi-scorer evaluation becomes necessary
* Evidence: 
  * .copilot-tracking/research/2026-07-25/b2-templates-implementation-research.md (Lines 300-365)
  * Data strategy research endorses config-driven for experimental repos
  * Only 2 of 7 transition criteria strongly met

### IP-01: Minimal Manifest Layer (Alternative 2)

* Approach: Create data/manifests/shared_v1/prompts.jsonl with structured provenance metadata and PromptDataset loader class
* Trade-offs:
  * Benefits: Structured provenance (prompt_id, source_family, semantic_tag), enables multi-scorer infrastructure, external contributions via JSONL
  * Drawbacks: 1-2 days implementation time, adds complexity without immediate payoff, requires data loading infrastructure not yet present
* Rejection rationale: Over-engineered for experimental repo. Multi-scorer evaluation (primary manifest benefit) is future work. Git version control sufficient for prompt management in experimental context.

### IP-02: Full Manifest-First Implementation (Alternative 3)

* Approach: Complete manifest-based architecture with Dataset abstraction, split management, offline rewards, no config tuple fallback
* Trade-offs:
  * Benefits: Production-ready, full provenance tracking, multi-scorer infrastructure ready
  * Drawbacks: 1-3 days implementation, massive over-engineering, significantly slows iteration, high maintenance burden
* Rejection rationale: Violates "experimental repo" project context. User explicitly stated "I don't need production deployment in this repo. I need to set it up for experiments with baselines." Transition criteria not met.

### IP-03: Hybrid (Templates in Code, Splits in Manifests) (Alternative 4)

* Approach: Store prompts as tuples in code, use JSONL files with split indices
* Trade-offs:
  * Benefits: Balances code convenience with split structure
  * Drawbacks: Confusing split between code and data, index-based splits fragile to reordering, no compelling use case
* Rejection rationale: Adds complexity without clear benefits. Either keep everything in code (Alternative 1, simpler) or move to full manifest structure (Alternative 2, more thorough).

## Suggested Follow-On Work

Items identified during planning that fall outside current scope.

* WI-01: Extract Templates 2 and 3 from B2-DiffuRL Appendix H — High priority, blocks Phase 2 completion
  * Source: Research gap identified in template specifications
  * Dependency: Access to B2-DiffuRL CVPR 2025 paper
  * Estimated effort: 2-4 hours
  * Timeline: Can proceed in parallel with Template 1 implementation or sequentially after Phase 1

* WI-02: Migrate to manifest-based architecture when transition criteria met — Medium priority, future work
  * Source: Data strategy phased evolution plan
  * Dependency: Multi-scorer evaluation becomes necessary OR publication requires detailed provenance
  * Estimated effort: 1-2 days
  * Trigger conditions documented in research (Lines 565-595)

* WI-03: Add template-specific reward analysis — Low priority, experimental enhancement
  * Source: Research scenario "which template improves most with RL?"
  * Dependency: Full B2 implementation completion, baseline training runs
  * Estimated effort: 4-8 hours for analysis scripts
  * Enables understanding of which semantic domains (activities vs attributes vs relations) benefit most from RL fine-tuning

* WI-04: Create benchmark comparison suite for cross-baseline evaluation — Medium priority, evaluation infrastructure
  * Source: Research emphasis on fair cross-baseline comparison
  * Dependency: All baselines trained on identical B2_FULL prompts
  * Estimated effort: 1 day for automated comparison scripts
  * Outputs: Reproducible ranking of DDPO vs DPOK vs B2-DiffuRL on identical prompts
