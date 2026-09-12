# T2I-CompBench in the evaluation notebook

Open `clover/exp/total_evals.ipynb`. The benchmark setup, prompt loading, image
generation, and metric aggregation functions are imported from
`clover/evaluate/t21_comp_bench.py`, next to `inference.py`.
`run_experiments.sh` is unchanged; no benchmark CLI or job submission is required.

Run the notebook's T2I-CompBench cells in order:

1. Execute the import cell for `clover.evaluate.t21_comp_bench`.
2. Call `setup_compbench()` to fetch/verify the official source and prompts.
3. On first use, run `setup_compbench(install_runtime=True)` to install the local
   metric runtime and UniDet weights. This requires a CUDA 11.7-compatible build
   toolchain for Detectron2. BLIP weights download on first scoring.
4. Load a checkpoint with the existing `load_baseline_model(...)` import, or use
   an already-loaded `model`.
5. Run `generate_compbench_images(model, ...)` with a new run name.
6. Offload `model.pipe` to CPU as shown, then call `run_compbench_metrics(manifest_path)`.
7. Inspect `results["summary"]`. Use `collect_compbench_metrics(manifest_path)` to
   reload existing metric scores without rerunning the metric models.

BLIP-VQA scores color, shape, and texture. UniDet scores spatial relationships.
All six validation prompt categories are available (300 prompts each), but
non-spatial and complex metrics are not included. Ten images per prompt across
the four supported categories produces 12,000 images. For a small check, use
`prompts_by_category={"color": compbench_color_prompts[:2]}` and
`images_per_prompt=1`.

Images/manifests are saved under
`clover/data/<baseline>/t2i_compbench/<run_name>/`; per-image scores and category
summaries are saved under
`outputs/<baseline>/evals/t2i_compbench/<run_name>/metrics.json`.

The pinned official source is downloaded from
[Karine-Huang/T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench), revision
`4aa404212eb5d06e5adbcd9cee696c750d0d25a5`. Its older dependencies remain in the
project-local uv runtime `clover/evaluate/compbench_backend/.venv`, separate from
the model-generation environment. Each scorer sees one GPU and runs sequentially.
Runtime installation, native compilation, and full GPU scoring have not been
validated; CPU tests exercise the imported module and result bookkeeping.
