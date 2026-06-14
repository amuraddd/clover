# Repo Data Surfaces Research

## Status
Complete

## Primary Topic
Current codebase structure and any existing data-related patterns relevant to building a shared dataset subset for diffusion-model RL baselines.

## Research Questions
- What repository structure exists for data ingestion, baselines, experiments, and utilities?
- What code, notebooks, or placeholders indicate expected dataset formats or workflows?
- What exact file references and line-level evidence support those observations?
- What notable gaps exist in dataset code, schemas, configs, or conventions?
- What are the implications for a shared baseline dataset subset?

## Findings

### Repository Structure Relevant to Data Work

- Top-level package structure is present but largely unimplemented: `clover/baselines/` and `clover/data/` are empty directories from repository inspection, while `clover/exp/` contains one notebook and `clover/utils/` contains one utility module.
- The repository intent is diffusion-model RL research rather than general application code. README.md:2 states the repo is for "training diffusion models using reinforcement learning" and README.md:5-9 enumerates four baseline paper references.
- The Python package metadata is still scaffold-level. pyproject.toml:4 uses the placeholder description `Add your description here`, pyproject.toml:7 declares `dependencies = []`, and no package-level dependency choices indicate an established data stack.
- The CLI entry point is only a stub. main.py:2 prints `Hello from clover!`, which indicates no current training, ingestion, or experiment orchestration path outside notebooks.

### Existing Data-Related Code and Workflow Signals

- The only checked-in executable workflow is a notebook bootstrap flow in clover/exp/b2diffurl.ipynb. The first code cell changes into the repository root at clover/exp/b2diffurl.ipynb:11, which suggests manual notebook-driven exploration rather than reusable dataset loaders or scripts.
- The second notebook cell imports `notebook_line_magic` from clover.utils.utils at clover/exp/b2diffurl.ipynb:21 and executes it at clover/exp/b2diffurl.ipynb:22. This is the only package integration currently exercised by the notebook.
- The notebook utility is solely for interactive autoreload support. clover/utils/utils.py:3 defines `notebook_line_magic()`, and clover/utils/utils.py:12-13 only reload the `autoreload` extension and enable `%autoreload 2`.
- There is no observed code for dataset loading, path resolution, prompt parsing, image decoding, manifest creation, subset sampling, reward annotation, or train/eval split generation anywhere in the checked-in package files returned by repository search.

### Planning Artifacts That Indicate Intended Dataset Work

- The main research note already frames dataset definition as a pending project need rather than an implemented system. .copilot-tracking/research/2026-06-14/data-strategy-research.md:8-9 explicitly asks to define dataset definitions and a shared baseline subset strategy.
- That same planning note assumes the repository is still in early setup. .copilot-tracking/research/2026-06-14/data-strategy-research.md:16 says the repository is at an early stage with directories already created, and line 17 says the first milestone is comparability across baselines.
- The sibling subagent note shows unresolved baseline-data assumptions. .copilot-tracking/research/subagents/2026-06-14/baseline-data-assumptions-research.md:16-17 still ask for the common cross-baseline subset and practical dataset candidates.
- These `.copilot-tracking` files are useful evidence of intended workflow boundaries, but they are planning documents only. They do not define a concrete on-disk schema, code API, or data contract yet.

### Notable Gaps

- No dataset code exists under `clover/data/`; the directory is empty.
- No baseline implementation code exists under `clover/baselines/`; the directory is empty.
- No configuration files define dataset roots, cache locations, licenses, prompt sources, image preprocessing rules, split conventions, or subset manifests.
- No dependency declarations exist for likely data tooling such as `datasets`, `pandas`, `pyarrow`, `Pillow`, `webdataset`, or `torch` ecosystem packages.
- No documented schema exists for prompt-only records, prompt-image pairs, generated outputs, reward labels, or evaluation artifacts.
- No reproducibility convention is visible for random seeds, subset IDs, split naming, or artifact versioning.

### Implications for a Shared Baseline Dataset Subset

- A shared baseline subset will need to start from a new explicit dataset contract because the repository currently has no existing contract to extend.
- The current notebook-oriented workflow suggests the first usable subset should favor a simple manifest-based format that works both interactively and from future scripts. A row-oriented manifest with stable IDs, prompt text, optional source image path, split label, and metadata fields would fit the present codebase better than a complex custom storage layer.
- Because `clover/baselines/` is empty, the subset format should be baseline-agnostic and avoid embedding assumptions specific to DDPO, DPOK, or sparse-reward fine-tuning until those implementations exist.
- The absence of dependencies and loaders means any shared subset decision should include artifact boundaries up front: raw source references, normalized manifest, optional cached tensors or latents, and evaluation outputs should be separated deliberately rather than inferred later.
- The strongest current constraint is comparability across baselines, as stated in .copilot-tracking/research/2026-06-14/data-strategy-research.md:17. That points toward building a smallest-common-denominator subset centered on prompt records and deterministic split definitions first, then layering baseline-specific derived artifacts later.

## Open Questions

- Which first-class data unit should the implementation optimize for: prompt-only generation tasks, prompt-image pairs, or prompt plus reward annotations?
- Will the initial shared subset be sourced from an external dataset provider, an internally curated prompt set, or generated outputs from an existing diffusion checkpoint?
- Are there licensing or redistribution constraints that prohibit storing raw images inside the repository-managed dataset subset?
- Which baseline evaluation outputs need to be shared across methods: generated images only, prompt-image-reward tuples, or richer rollout traces?

## Recommended Next Research

- Confirm the paper-specific data assumptions for each referenced baseline and map them to a common minimal record schema.
- Decide whether the first subset should be prompt-only or prompt-image paired based on the baselines selected for first implementation.
- Define artifact boundaries and storage locations outside the repository tree if large data will not live in Git.
- Evaluate one concrete manifest format, such as JSONL or Parquet, against the expected training and notebook workflows.
- Establish naming conventions for subset versioning, split IDs, and deterministic sampling before any baseline code is added.
