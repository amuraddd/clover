# Diagram Benchmark Research

## Status

Complete for the requested scope.

## Research Scope

- Paper: From Words to Structured Visuals: A Benchmark and Framework for Text-to-Diagram Generation and Editing.
- Venue/version: CVPR 2025, DOI 10.1109/CVPR52734.2025.01243, arXiv:2411.11916.
- Focus: data-strategy implications for a prompt-diagram dataset track, including graphs, flowcharts, and prompt-to-code-to-diagram generation.

## Questions

- What benchmark datasets or tasks does the paper define?
- Which diagram categories are most relevant to directed graphs, undirected graphs, and flowcharts?
- What should the canonical raw data unit be for a prompt-diagram or prompt-code-diagram dataset?
- Does the paper suggest or align with intermediate code or structured representations?
- What schema fields should store prompt, structured spec or code, rendered diagram, and evaluation metadata?
- What are the implications for a modular codebase with shared dataset abstractions in clover/data and baselines in clover/baselines?

## Key Findings

### 1. What benchmark datasets and tasks the paper defines

The paper defines one benchmark, DiagramGenBenchmark, with three task views over the same underlying diagram corpus:

- Diagram generation: natural-language instruction to diagram code, then compiled diagram.
- Diagram coding: diagram image to compilable diagram code.
- Diagram editing: original diagram or code plus an edit instruction to modified code and modified diagram.

Authoritative evidence:

- Main text, Section 5: DiagramGenBenchmark focuses on transforming textual descriptions into structured diagram representations and includes model architecture diagrams, flowcharts, line charts, directed and undirected graphs, tables, bar charts, and mind maps.
- Appendix A.2 states: “We constructed three distinct tasks” and then enumerates diagram generation, diagram coding, and diagram editing.
- Section 3 formalizes the three task functions:
	- Diagram generation: $D_{gen} = f_{img}(f_{code}(x_{ins}))$
	- Diagram coding: $c'_{diag} = f^{-1}_{code}(D_{gen})$
	- Diagram editing: $D_{mod} = f_{img}(f_{mod}(f^{-1}_{code}(D_{ori}), x_{edit}))$

Dataset sizes reported by the paper:

- Diagram generation / coding: 6,713 training instances and 270 test instances.
- Diagram editing: 1,400 training instances and 200 test instances.
- Appendix A.1 reports 13,000+ raw code samples filtered to 6,983 compilable code-image pairs.

Interpretation for Clover:

- Treat “benchmark” as a shared corpus plus task-specific projections, not as three unrelated datasets.
- The benchmark is code-centric and compiler-backed, with prompts and edit instructions derived from the same base artifacts.

### 2. Diagram categories relevant to directed graphs, undirected graphs, and flowcharts

The benchmark defines eight diagram categories:

- model architecture diagram
- flowchart
- line chart
- directed graph
- undirected graph
- table
- bar chart
- mind map

Most relevant categories for a prompt-diagram track centered on graphs and flow control are:

- Directed graph: exact match for graph-structured diagrams with edge direction.
- Undirected graph: exact match for symmetric relation graphs.
- Flowchart: exact match for procedural and decision-flow diagrams.
- Model architecture diagram: highly relevant because many architecture diagrams are directed acyclic or layered graphs with typed nodes/edges.
- Mind map: relevant for tree-like or radial graph structures, though semantics differ from general graph tasks.

Less central for this track:

- line chart
- bar chart
- table

Dataset counts from Table 1 / Section 5.1:

- Flowchart: 1,500 generation-coding samples; 320 editing samples.
- Directed graph: 787 generation-coding; 217 editing.
- Undirected graph: 73 generation-coding; 18 editing.
- Model architecture: 3,000 generation-coding; 706 editing.
- Mind map: 140 generation-coding; 16 editing.

Implication:

- A Clover graph/flowchart track should expect class imbalance if it mirrors the paper. Undirected graph coverage is especially thin and should likely be augmented if graph topology robustness matters.

### 3. Canonical raw data unit for a prompt-diagram or prompt-code-diagram dataset

The paper’s raw curation pipeline starts from code artifacts, not prompts:

- Appendix A.1: raw Tex and DOT language-based code were gathered from VGQA, datikz, datikz-v2, GitHub, and Overleaf.
- Each sample was compiled, and only compilable examples were kept.
- The result was a corpus of code-image pairs.
- Appendix A.2: human-like instructions for diagram generation were then reverse-engineered from existing open-source diagram code using GPT-4o.
- Editing examples were derived by generating modification suggestions and applying them to original code, then retaining unique compilable results.

Recommendation:

- The canonical raw data unit for Clover should be a compilable diagram artifact, not just a prompt-image pair.
- Concretely, the base unit should be:
	- one structured source spec or code artifact
	- one deterministic renderer or compiler configuration
	- one rendered diagram output
	- category and provenance metadata

Then task-specific annotations should be layered on top:

- prompt(s) for generation
- image-to-code view for coding
- edit instruction plus parent-child lineage for editing

Reasoning:

- This matches the paper’s actual data generation pipeline.
- It preserves reversibility between code and image, which is required for coding and editing tasks.
- It supports prompt-to-code-to-diagram workflows better than treating the prompt as the only primary object.

Recommended canonical unit abstraction:

- diagram artifact record
	- immutable identity of the source diagram spec
	- compiled render outputs
	- derived prompt annotations
	- optional edit descendants

### 4. Whether the paper suggests or is compatible with intermediate code or structured representations

Yes. The paper is strongly compatible with intermediate code and, in practice, depends on it.

Evidence:

- Section 3 defines diagram generation as instruction to code to image.
- Section 3 defines diagram coding as image to code.
- Section 3 defines diagram editing as image to code, then code modification, then recompilation.
- Section 5 states the repositories predominantly feature diagram code written in LaTeX or DOT languages.
- Appendix B describes prompts for Diagram-to-Code Agent handling conversion of diagrams to LaTeX or DOT code.
- Section 4.4 says the Check Agent compiles generated code and uses compiler feedback for debugging.

Implication:

- Intermediate structured representations are not optional in this benchmark design; they are the core supervision target and the backbone for evaluation.
- For Clover, a prompt-code-diagram dataset should explicitly support multiple structured backends such as DOT, TikZ/LaTeX, Mermaid, or other declarative diagram DSLs rather than flattening everything into raster images.

### 5. Recommended schema fields

Recommended base artifact schema:

- diagram_id: stable identifier for the base artifact.
- split: train, validation, test, or custom split.
- category: one of the supported diagram classes.
- subcategory: optional finer tag such as DAG, workflow, pipeline, tree, network, architecture.
- source_language: DOT, TikZ, LaTeX, Mermaid, Graphviz, etc.
- source_code: canonical code or structured spec.
- normalized_spec: optional AST, JSON graph, or normalized IR if Clover introduces one.
- compiler_or_renderer: exact renderer used.
- compiler_version: versioned rendering environment.
- render_config: flags, fonts, packages, layout engine, seed if relevant.
- rendered_path or rendered_blob_ref: pointer to the rendered output.
- rendered_format: pdf, png, svg, jpg.
- provenance_source: VGQA, datikz, datikz-v2, GitHub repo, Overleaf, or manual.
- provenance_uri: original URL or repository reference.
- license: upstream license.
- compile_success: whether the source compiles or renders.
- compile_logs: compiler stderr or diagnostics.

Recommended prompt/task annotation schema:

- task_type: generation, coding, editing.
- prompt_id: stable identifier for each prompt or instruction.
- prompt_text: natural-language instruction.
- prompt_origin: human-authored, reverse-engineered, synthetic, edited, paraphrased.
- complete_prompt_text: expanded or normalized instruction if query expansion is used.
- input_image_ref: for coding or editing tasks.
- parent_diagram_id: for edit tasks.
- edit_instruction: requested modification text.
- target_code: expected output code for the task.
- target_render_ref: expected rendered diagram.
- derivation_method: GPT-4o reverse engineering, model-assisted edit creation, human annotation, etc.
- uniqueness_hash: to deduplicate near-identical code or rendered outputs.

Recommended evaluation metadata schema:

- pass_at_1
- rouge_l
- codebleu
- edit_distance
- chrf
- ruby
- clip_fid
- lpips
- psnr
- ms_ssim
- human_score_mean
- human_score_count
- evaluator_protocol_ref
- error_tags: shape, structure, content, position, layout, color, line_type, font

Why this schema fits the paper:

- The paper evaluates both code correctness and rendered-diagram quality.
- The benchmark includes both forward and reverse transformations.
- The error analysis explicitly identifies reusable error taxonomies.

### 6. Implications for a modular Clover codebase

The current repository layout shows empty or not-yet-populated data and baseline subtrees under clover/data and clover/baselines. Given the paper, the clean modular split is:

- clover/data
	- shared dataset abstractions for diagram artifact, task projection, split logic, schema validation, provenance, and metric records
	- renderer/compiler interface definitions and normalization utilities
	- category taxonomy and task registry
	- loaders that can expose the same base artifact as generation, coding, or editing samples
- clover/baselines
	- task-specific baselines that consume the shared abstractions
	- prompt-to-code baselines
	- image-to-code baselines
	- edit-code baselines
	- agentic baselines that add planning, checking, or iterative repair on top of shared data interfaces

Concrete design implication from the paper:

- Do not build separate raw dataset formats for prompt-to-diagram, image-to-code, and edit tasks.
- Build one compiler-backed artifact store and expose multiple task adapters.

This is the paper’s underlying pattern:

- same code-image corpus
- multiple task formulations
- shared metrics across code quality and image quality

Recommended Clover abstraction boundary:

- Base artifact: structured diagram specimen with source code, renderer, render, provenance.
- Task view: generation view, coding view, editing view, each pointing back to the same artifact or artifact lineage.
- Baseline contract: any baseline returns both structured output and rendered output when possible, so evaluation can remain benchmark-aligned.

### 7. Exact citations and authoritative references

Primary paper references:

- CVPR 2025 DOI: [https://doi.org/10.1109/CVPR52734.2025.01243](https://doi.org/10.1109/CVPR52734.2025.01243)
- IEEE Xplore landing page from Crossref metadata: [https://ieeexplore.ieee.org/document/11093195/](https://ieeexplore.ieee.org/document/11093195/)
- arXiv abstract: [https://arxiv.org/abs/2411.11916](https://arxiv.org/abs/2411.11916)
- arXiv HTML full text used for section extraction: [https://arxiv.org/html/2411.11916v1](https://arxiv.org/html/2411.11916v1)

Specific paper-backed statements used above:

- Abstract: benchmark has eight diagram categories and DiagramAgent has Plan Agent, Code Agent, Check Agent, Diagram-to-Code Agent.
- Section 3: formal task decomposition into generation, coding, and editing through code intermediates.
- Section 5 and Table 1: category list and dataset statistics.
- Appendix A.1: raw data sources are Tex and DOT code from VGQA, datikz, datikz-v2, GitHub, and Overleaf; 13,000+ raw samples filtered to 6,983 compilable code-image pairs.
- Appendix A.2: task construction details, including reverse-engineered instructions and edit-suggestion generation.
- Appendix B: prompts explicitly target LaTeX or DOT code.
- Appendix C / Table 8: benchmark comparison highlights input-output formats and task specialization.
- Appendix F: reusable error taxonomy for evaluation metadata.

## Recommended Next Research

- Inspect whether the authors released DiagramGenBenchmark publicly and whether the released files preserve prompt, code, render, and edit lineage separately or only as task-specific exports.
- Compare this benchmark with AutoTikZ, Plot2Code, MatPlotBench, and Design2Code to decide whether Clover needs one unified diagram IR across charts and graphs or separate IR families.
- Validate whether Mermaid should be a first-class source language in Clover or treated as a normalization target from DOT and TikZ.
- Check licensing compatibility in the upstream sources named by the paper before mirroring any data artifacts.

## Open Uncertainties

- The paper clearly states the data sources and task construction method, but it does not fully specify a released file-level schema in the text available through arXiv HTML.
- The paper names LaTeX and DOT as dominant source languages; it does not claim a language-agnostic intermediate representation beyond code itself.
- Undirected graph coverage is small, so the paper alone is not enough to justify robust generalization claims for that category.