# DreamSim diversity in B2 evaluation

`b2_dreamsim_worker.py` loads the official pretrained DreamSim ensemble and
its RGB, 224 × 224 preprocessing in a separate, locked uv runtime, following
the ImageReward subprocess pattern. The main training environment is unchanged.
`score_b2_dreamsim(plan_path, device="cuda:0", batch_size=8)` reads the existing
B2 image manifests. `run_b2_evaluation()` calls it after ImageReward and before
summarization. No extra images are generated.

## Metric definition

[Section 4.2 of the supplied ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/ec52962a55a83c3d860b5c9b7338422b-Paper-Conference.pdf)
describes diversity as variance of image features and uses batches of 64.
The text does not specify a variance correction or reduction over feature
dimensions. This implementation explicitly chooses the **sum of sample
variances** of the normalized [DreamSim ensemble embeddings](https://github.com/ssundaram21/dreamsim):

```text
e_i = L2_normalize(DreamSim.embed(preprocess(image_i)))
mean_e = mean_i(e_i)
diversity = sum_i ||e_i - mean_e||² / (n - 1)
```

DreamSim's own embedding centering/normalization remains enabled. For unit
embeddings, this statistic equals the mean cosine distance over all distinct
image pairs. Higher values indicate greater diversity; identical features
give zero. The implementation embeds each image once, then computes variance
without allocating a pairwise distance matrix.

Images are grouped by `(baseline, seed, template, prompt_index)`. Different
prompts, baselines and evaluation seeds are never pooled. B2's configured
`images_per_prompt` determines `n`; it is not forced to the paper's 64.
At least two images per group are required. `batch_size` controls only feature
extraction memory and does not change the statistical grouping. Results are
stored in raw units, without the paper table's display scaling.

## Saved results

- `outputs/<baseline>/evals/b2/<run>/dreamsim.json`: cached image records plus
  model, estimator, grouping and runtime-lock metadata. Each record contains
  `dreamsim_diversity` (its group's score), `dreamsim_image_count`, and
  `dreamsim_contribution` (`n/(n-1)` times its squared distance to the centroid).
- `clover/evaluate/metrics/<run>/b2_dreamsim.json`: stage export for all baselines.
- B2 per-image, per-seed/template and across-seed summary files include the new
  metric. Prompt scores have equal weight; across-seed summaries report the
  mean and sample standard deviation. A single seed's standard deviation is
  JSON `null`, consistent with existing metrics.
- Requests, worker results and logs live under
  `clover/data/b2_evaluation/<run>/dreamsim_{request,results}.json` and
  `dreamsim.log`. Model weights are cached in `.cache/DreamSim`.

The helper installs its runtime with `uv sync --locked --project
clover/evaluate/b2_evals/b2_dreamsim_backend` if needed. Pretrained weights are
downloaded on the first scoring run. It exposes at most one GPU to the worker
and reuses valid scores bound to the plan, image checksums and runtime lock.
Historical summaries without DreamSim remain readable; partial DreamSim
results across baselines are rejected.

Run experiments through the project's authorized Slurm launcher,
`sbatch run_experiments.sh`. Runtime installation and CPU unit checks do not
launch an experiment.
