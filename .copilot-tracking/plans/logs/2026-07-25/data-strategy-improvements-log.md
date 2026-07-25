<!-- markdownlint-disable-file -->
# Planning Log: Data Strategy Improvements

## Discrepancy Log

Gaps and differences identified between research findings and the implementation plan.

### Unaddressed Research Items

* DR-01: Full manifest-based data architecture with JSONL/Parquet prompt manifests
  * Source: .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 444-582)
  * Reason: Research explicitly states "No immediate action required" and recommends this only when transition criteria are met (prompt scaling >20, cross-baseline comparison needs, multi-scorer evaluation)
  * Impact: Low - Current config-driven approach is sufficient for ongoing work per research recommendation
* DR-02: Multi-scorer reward evaluation (CLIP, BERT, ImageReward)
  * Source: .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 675-708)
  * Reason: Research classifies this as "Phase 2: Multi-Scorer Evaluation" in transition roadmap, not immediate requirement
  * Impact: Low - Single aesthetic reward is adequate for baseline bring-up and method validation
* DR-03: Structured diagrams dataset family (flowcharts, graphs)
  * Source: .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 710-737)
  * Reason: Research classifies this as "Phase 3: Structured Diagrams" in transition roadmap, separate track from RL prompts
  * Impact: Low - Not applicable to current RL baseline milestone
* DR-04: Deterministic train/eval split manifests
  * Source: .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 458-468)
  * Reason: Plan implements lightweight split via separate eval_prompts tuples; full manifest splits deferred per research staging
  * Impact: Medium - Partial implementation addresses immediate need for eval methodology; full split enforcement deferred
* DR-05: B2-DiffuRL three-template prompt suite as canonical raw subset
  * Source: .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 472-490)
  * Reason: Research identifies this as the ideal common subset for manifest-based approach; plan continues with simple clover prompts for config-driven phase
  * Impact: Medium - Plan maintains current simple prompts; B2 templates deferred to manifest-based evolution

### Plan Deviations from Research

* DD-01: Plan uses simple eval_prompts tuples instead of manifest-based split files
  * Research recommends: Deterministic split manifests (split_train.jsonl, split_eval_in_domain.jsonl, split_eval_generalization.jsonl)
  * Plan implements: config.eval_prompts tuple separate from config.train_prompts
  * Rationale: Research explicitly stages manifest-based splits as "Phase 1: Minimal Manifest Layer (1-2 weeks)" in transition roadmap; current plan implements quick wins that maintain config-driven simplicity while establishing split boundaries conceptually
* DD-02: Plan maintains clover-themed prompts instead of importing B2-DiffuRL template suite
  * Research recommends: Use B2-DiffuRL three-template suite as canonical raw subset with Template 1 as DDPO/B2 overlap slice
  * Plan implements: Shared DEFAULT_TRAIN_PROMPTS and DEFAULT_EVAL_PROMPTS in clover/utils/prompts.py using existing clover themes
  * Rationale: B2 templates are appropriate for manifest-based architecture with provenance tracking; current plan maintains continuity with existing experiments and avoids premature complexity
* DD-03: Plan adds reward_type config field but only registers aesthetic reward
  * Research recommends: Multi-scorer infrastructure with CLIP, BERT, ImageReward, Aesthetic
  * Plan implements: REWARD_REGISTRY pattern with only "aesthetic" entry, commented placeholders for future scorers
  * Rationale: Research classifies multi-scorer as "Phase 2: Multi-Scorer Evaluation" requiring offline reward computation; current plan establishes extensible pattern while deferring scorer implementation per staging

## Implementation Paths Considered

### Selected: Optional Improvements to Config-Driven Approach

* Approach: Enhance current config-driven architecture with prompt deduplication, explicit eval splits, and configurable reward functions while maintaining simplicity and self-contained reproducibility
* Rationale: Research explicitly recommends "Continue with config-driven approach for ongoing work" and states "No immediate action required" unless transition criteria are met; selected path provides incremental value without disrupting working baselines
* Evidence: .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 643-668: "Immediate Actions (Current State Sufficient)")

### IP-01: Full Manifest-Based Architecture Implementation

* Approach: Implement complete manifest-based data strategy with JSONL prompts, deterministic splits, B2-DiffuRL template suite, multi-scorer evaluation, and structured diagrams track
* Trade-offs:
  * Benefits: Full provenance tracking, deterministic splits, cross-baseline comparison, multi-scorer support, scalability to hundreds of prompts
  * Drawbacks: Significant complexity increase, requires data pipeline infrastructure, premature for current needs, disrupts working experiments
* Rejection rationale: Research transition criteria not yet met - prompt sets still small (~4 prompts), single reward function sufficient, no cross-baseline comparison requirements yet; research explicitly stages this as multi-phase evolution (Phase 1-4) triggered by concrete needs

### IP-02: Minimal Manifest Layer Only (Phase 1)

* Approach: Implement Phase 1 only from research transition roadmap - create PromptDataset class, JSONL manifests, optional loading with config fallback
* Trade-offs:
  * Benefits: Establishes data abstraction without full commitment, maintains backward compatibility, enables gradual migration
  * Drawbacks: Adds infrastructure before it provides clear value, splits effort between two approaches, may never get used if transition criteria aren't met
* Rejection rationale: Research identifies this as appropriate when "prompt sets grow beyond 20 prompts" or "need deterministic train/eval splits across runs"; current 4-prompt sets don't justify the infrastructure yet; selected path (IP-Selected) provides split boundaries via eval_prompts without requiring manifest files

### IP-03: Do Nothing, Maintain Status Quo

* Approach: Make no changes to current implementation, accept prompt duplication and lack of eval split
* Trade-offs:
  * Benefits: Zero implementation risk, no changes to working code, maximum simplicity
  * Drawbacks: Prompt maintenance burden continues, eval methodology remains weak, no progress toward future evolution
* Rejection rationale: Research identifies "Optional Improvements" with clear incremental value; selected path provides low-risk enhancements that improve maintainability and prepare architectural patterns for future evolution without premature complexity

## Suggested Follow-On Work

Items identified during planning that fall outside current scope.

### Transition Roadmap (Implement When Criteria Are Met)

* WI-01: Phase 1 - Minimal Manifest Layer (1-2 weeks) — High priority when triggered
  * Source: .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 670-682)
  * Trigger: Prompt sets grow beyond 20 prompts OR need deterministic train/eval splits across runs OR need cross-baseline comparison
  * Dependency: Current plan completion
  * Tasks:
    * Create clover/data/prompts.py with PromptDataset class
    * Create data/manifests/shared_v1/prompts.jsonl with B2 Template 1, 2, 3
    * Update baseline configs to optionally load from prompt_manifest_path
    * Add --prompt-manifest CLI arg
* WI-02: Phase 2 - Multi-Scorer Evaluation (2-3 weeks) — High priority when triggered
  * Source: .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 684-708)
  * Trigger: Need to compare CLIP vs BERT vs ImageReward vs Aesthetic
  * Dependency: WI-01 completion (manifest infrastructure)
  * Tasks:
    * Materialize prompt manifest for clover_shared_rl_subset_v1
    * Encode deterministic split files (train, eval_in_domain, eval_generalization)
    * Define run manifest schema for generated outputs and reward traces
    * Implement multi-scorer support in rewards_utils.py (CLIP, BERT, ImageReward, Aesthetic)
    * Create offline reward computation script
    * Update evaluation to load and compare multi-scorer results
* WI-03: Phase 3 - Structured Diagrams (4-6 weeks) — Medium priority, parallel to WI-02
  * Source: .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 710-737)
  * Trigger: Start of diagram generation benchmarks for graphs/flowcharts
  * Dependency: None (separate track from RL prompts)
  * Tasks:
    * Materialize diagram artifact schema for clover_structured_diagrams_v1
    * Start with flowchart and directed_graph categories
    * Standardize prompt-to-code renderer toolchain (Graphviz, TikZ)
    * Create clover/data/diagrams.py with DiagramDataset class
    * Add evaluation metrics (CodeBLEU, CLIP-FID)
* WI-04: Phase 4 - Full Manifest-Based Architecture (ongoing) — Medium priority when triggered
  * Source: .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 739-755)
  * Trigger: Multiple phases completed, project standardizes on manifest-first approach
  * Dependency: WI-01, WI-02, WI-03 completion
  * Tasks:
    * Define clover/data/base.py with abstract Dataset interface
    * Implement PromptDataset, DiagramDataset, PairedImageDataset concrete classes
    * Add versioning and dataset cards to manifests
    * Deprecate config tuple approach, require manifest paths
    * Add data preprocessing and validation scripts
    * Document dataset creation guide

### Transition Criteria Monitoring

* WI-05: Monitor prompt scaling — Low effort, ongoing
  * Source: .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 757-781)
  * Trigger: When prompt sets approach 20 prompts
  * Dependency: None
  * Action: Evaluate whether to trigger WI-01 (Phase 1 implementation)
* WI-06: Monitor cross-baseline comparison needs — Low effort, ongoing
  * Source: .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 757-781)
  * Trigger: When need to run fair comparisons across DDPO/DPOK/B2-DiffuRL with identical prompt sequences
  * Dependency: None
  * Action: Evaluate whether to trigger WI-01 (Phase 1 implementation)
* WI-07: Monitor reward function diversity needs — Low effort, ongoing
  * Source: .copilot-tracking/research/2026-06-14/data-strategy-research.md (Lines 757-781)
  * Trigger: When single aesthetic reward becomes insufficient for research questions
  * Dependency: None
  * Action: Evaluate whether to trigger WI-02 (Phase 2 implementation)

### Additional Enhancements

* WI-08: Add reward scorer implementations (CLIP, BERT, ImageReward) — Medium priority
  * Source: Research multi-scorer recommendations and REWARD_REGISTRY extensibility
  * Trigger: When WI-02 (Phase 2) is triggered
  * Dependency: Current plan completion (REWARD_REGISTRY pattern established)
  * Tasks:
    * Implement clip_reward in rewards_utils.py using CLIP model
    * Implement bert_reward in rewards_utils.py using BERT-based scorer
    * Implement imagereward_score in rewards_utils.py using ImageReward model
    * Add model loading and caching logic
    * Update REWARD_REGISTRY with new entries
    * Test reward_type config with multiple scorers
* WI-09: CLI arg for --reward-type override — Low priority
  * Source: Extensibility of parse_config pattern in common.py
  * Trigger: When WI-08 completed and multiple scorers available
  * Dependency: WI-08 completion
  * Tasks:
    * Add --reward-type argument to parse_config in common.py
    * Test CLI override: python -m clover.baselines.ddpo --reward-type clip
    * Document in usage guide
* WI-10: Experiment reproducibility documentation — Low priority
  * Source: Research emphasis on config.json reproducibility
  * Trigger: After first paper submission using baselines
  * Dependency: Current plan completion
  * Tasks:
    * Document how to reproduce experiments from config.json + seed
    * Document prompt provenance once manifest-based (WI-01 complete)
    * Create reproducibility checklist for paper submissions
