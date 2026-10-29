# Experimental Setup and Prompt Splits

This document describes the current EMO v5 training setup and its periodic evaluation, as configured on 2026-10-24. Settings are taken from [main.py](../../../main.py), [emo_v5.py](../emo_v5.py), and the shared [prompt definitions](../../utils/prompts.py). These describe the configured protocol; individual completed runs should be identified by their saved `config.json` and metrics. The objective and variance-head derivations are in [emo_v5_readme.md](emo_v5_readme.md).

## Prompt templates

The shared prompt pool contains **215 unique prompts** across three semantic families. Templates test activity composition, color binding, and spatial relationships.

| Template | Pattern | Construction | Examples |
| --- | --- | --- | --- |
| 1: Animal–activity | `a(n) [animal] [activity]` | Cartesian product of 45 animals and 3 activities: `riding a bike`, `playing chess`, and `washing dishes` | `a dog riding a bike`; `an elephant playing chess` |
| 2: Color–attribute | `[color] [fruit/vegetable]` | Explicit list of 40 color–object combinations; not a full Cartesian product | `red apple`; `brown banana`; `pink pomegranate` |
| 3: Spatial relation | `[object_1] [predicate] [object_2]` | Explicit list of 40 object–relation combinations using `under`, `on`, `on the left of`, and `on the right of` | `chair under umbrella`; `laptop on table`; `dog on the right of vase` |

Template 1 selects `a` or `an` from the animal name's initial letter. Templates 2 and 3 use the exact strings listed in the source file without adding articles or additional style instructions.

## Training and held-out evaluation split

**150 prompts (69.77%) are available for training, and 65 prompts (30.23%) are held out for evaluation.** These percentages use all 215 unique prompts as the denominator. The two sets have no exact prompt overlap.

| Template | Total prompts | Training prompts | Training % within template | Held-out prompts | Held-out % within template |
| --- | ---: | ---: | ---: | ---: | ---: |
| Animal–activity | 135 | 90 | 66.67% | 45 | 33.33% |
| Color–attribute | 40 | 30 | 75.00% | 10 | 25.00% |
| Spatial relation | 40 | 30 | 75.00% | 10 | 25.00% |
| **All templates** | **215** | **150** | **69.77%** | **65** | **30.23%** |

For Template 1, the first 30 animals in the source list supply all 90 training prompts, and the remaining 15 animals supply all 45 evaluation prompts. Animals are therefore disjoint across the splits, while the three activities are shared. For Templates 2 and 3, the first 30 listed prompts are used for training and the final 10 for evaluation. These splits hold out complete prompt strings; they do not guarantee disjoint colors, objects, or predicates. In particular, the held-out spatial prompts all use left/right relations, so this split is not balanced across relation types.

EMO v5 calls `get_prompts(seed=123, save=True)`. The seed shuffles the already-defined training and evaluation sets independently using one local random generator; it does not change split membership. The same prompt split and ordering are therefore used for both training seeds. The function saves the template, split, and prompt columns to `outputs/prompt_df.csv`.

Training draws 256 prompts per epoch uniformly **with replacement** from the 150-prompt training pool. Thus, 256 denotes generated trajectories, not distinct prompts, and a training epoch need not cover all 150 prompts. A complete 25-epoch run collects 6,400 fresh trajectories; replay adds update samples without changing the training prompt pool.

## Prompts actually used in periodic evaluation

The full held-out pool has 65 prompts, but EMO v5's periodic evaluation calls `standard_eval_prompts(config)` with its default limit of **10**. The helper samples without replacement using `random.Random(123)`. With the default prompt ordering, it selects the same 10 prompts at every evaluation event and for both training seeds.

| Periodic evaluation coverage | Count | Percentage |
| --- | ---: | ---: |
| Selected prompts out of the held-out pool | 10 / 65 | 15.38% |
| Selected prompts out of the complete prompt pool | 10 / 215 | 4.65% |
| Held-out prompts not used by periodic evaluation | 55 / 65 | 84.62% |

The selected subset contains 7 animal–activity, 1 color–attribute, and 2 spatial-relation prompts:

```text
a squirrel washing dishes
an ant playing chess
building on the right of truck
a rat washing dishes
pink pomegranate
car on the right of car
a bee washing dishes
an ant riding a bike
a bee playing chess
a deer playing chess
```

Periodic evaluation generates one image per selected prompt with image-generation seed 123. Evaluation runs every two completed epochs: epochs 2, 4, …, 24 in an uninterrupted 25-epoch run. This gives 12 events and 120 generated evaluation images, covering the same 10 distinct prompts. There is no additional automatic evaluation at epoch 25 under these settings.

The standalone [evaluate.py](../../../evaluate.py) benchmark runner has a separate B2 and T2I-CompBench sampling protocol. The 10-prompt counts above describe periodic training evaluation only and should not be attributed to those standalone benchmarks.

## Model and optimization configuration

The standard launch path is `sbatch run_experiments.sh`, which invokes `main.py` in the project's existing environment managed with `uv` and requests two GPUs. The active baseline in `main.py` is EMO v5. The following are the effective standard-launcher settings, including inherited EMO v5 defaults; direct module defaults differ for some parameters.

| Component | Setting |
| --- | --- |
| Base model | `runwayml/stable-diffusion-v1-5` |
| Training seeds | 123 and 456 |
| Image resolution | 512 × 512 |
| Denoising | 50 DDPM steps, learned-range variance, `eta=1.0` |
| Classifier-free guidance | 5.0, applied to noise prediction only |
| Negative prompt | `blurry, low quality, distorted` |
| Trainable parameters | UNet LoRA adapters and FP32 variance-output convolution |
| LoRA | Rank 32, alpha 32, dropout 0; targets `to_v`, `to_k`, `to_q`, `to_out.0` |
| Training duration | 25 epochs, 256 fresh trajectories per epoch |
| Updates | 3 passes over the combined fresh/replay batch per epoch; minibatch size 64 |
| Optimizer | AdamW; initial learning rate 1e-4; betas (0.9, 0.999); epsilon 1e-8 |
| Learning-rate schedule | Cosine warm restarts every 15 epochs, minimum 3e-6 |
| Maximum gradient norm | 0.1 |
| Training reward | CLIP reward |
| Reward scale / discount | 4.0 / 0.9 |
| Likelihood scale | 1.0; log probabilities averaged over latent dimensions |
| Importance weights | Detached and clipped to [0.9999, 1.0001] |
| Cross-entropy coefficient | 0.10; frozen snapshot refreshed at each `sac_update` |
| Entropy coefficient | Effective weight 1; configured `entropy_scale=0.20` is currently inactive |
| Replay | Diversity threshold 0.5; buffer reset after every 15 completed epochs |
| Memory settings | Mixed precision and gradient checkpointing enabled; rollout chunk size 64 |
| Checkpoint / evaluation intervals | Every 5 / 2 epochs |

Replay accepts a saved trajectory only if its prompt matches a fresh prompt, its reward is at least the fresh same-prompt mean, and its normalized Inception feature-distance score exceeds the diversity threshold. All fresh trajectories remain in the update batch. Advantages use same-prompt baselines estimated only from fresh samples, excluding the target sample itself when applicable.

The separate parameter grid in [emo_v5_experiments.py](../../../emo_v5_experiments.py) varies reward scale over {5, 10, 20}, cross-entropy coefficient over {0.01, 0.05, 0.1}, and diversity threshold over {0.25, 0.5, 0.75}. With two training seeds, this defines 54 combinations before completed-run exclusions. This grid is distinct from the fixed standard configuration above.

## Evaluation

Evaluation has two levels: periodic CLIP/BERT monitoring during training, described above, and standalone B2/T2I-CompBench benchmarking. The standalone runner's configured baselines are `ddpo`, `b2diffurl`, `emo_v5`, and the pretrained `sd15` reference. It loads trained checkpoints from **training seed 123**, then evaluates with **generation seeds 123, 124, and 126**. Variation across these generation seeds is therefore not variation across independently trained checkpoints.

**Current execution detail:** although `evaluate.py` contains both benchmark workflows, its call to `run_b2_evaluation` is commented out. The current entry point runs CompBench; the B2 settings below describe the implemented B2 workflow and configured defaults, not evidence that a particular invocation executed it.

### B2 prompt templates and sampling

B2 evaluation uses the same three held-out template families defined earlier, with the following names in its result files:

| Template | Result label | Held-out pool | Selected per generation seed | Fraction of held-out template |
| --- | --- | ---: | ---: | ---: |
| `a(n) [animal] [activity]` | `object_behavior` | 45 | 36 | 80% |
| `[color] [fruit/vegetable]` | `object_attribute` | 10 | 8 | 80% |
| `[object_1] [predicate] [object_2]` | `positional_relationship` | 10 | 8 | 80% |
| **Total** | | **65** | **52** | **80%** |

For each generation seed, the sampler selects `floor(0.8 × template_size)` prompts without replacement within each template. Baselines share the same prompt and image-seed manifest for paired comparisons. Selection may overlap between generation seeds; 52 × 3 does not represent 156 distinct prompt strings. The 52 selected prompts constitute 24.19% of the complete 215-prompt pool and come exclusively from its held-out portion.

The runner specifies **20 images per selected prompt**, giving **1,040 images per baseline per generation seed**, or **3,120 image records per baseline across three seeds**. Each replicate receives a deterministic image seed derived from the generation-seed/template/prompt/replicate identifiers. B2 inference uses 512 × 512 images, 50 denoising steps, guidance scale 7.5, and the negative prompt `blurry, low quality, distorted`.

### B2 evaluation metrics

| Metric | Computation in this repository | Interpretation |
| --- | --- | --- |
| CLIP similarity | Raw cosine similarity between normalized image and prompt embeddings from OpenCLIP `ViT-H-14`, pretrained with `laion2b_s32b_b79k`; not multiplied by 100 | Higher indicates stronger image–text alignment |
| BERTScore | BLIP (`Salesforce/blip-image-captioning-base`) first captions each image; `microsoft/deberta-large-mnli` compares the caption with the original prompt, without baseline rescaling | Higher precision, recall, and F1 indicate stronger caption–prompt semantic agreement; summary tables use F1 |
| ImageReward | `ImageReward-v1.0` scores each prompt–image pair | Higher indicates greater predicted human preference |
| FID | Fréchet distance between baseline and pretrained `sd15` image distributions, computed separately for each generation seed and template | Lower indicates closer agreement with the SD 1.5 reference distribution |
| DreamSim diversity | Trace of the sample covariance of normalized DreamSim ensemble embeddings across images generated for the same prompt | Higher indicates greater within-prompt perceptual variation |

For \(n\) generated images with normalized DreamSim embeddings \(z_i\), the diversity statistic is

$$
\operatorname{Diversity}
=\frac{1}{n-1}\sum_{i=1}^{n}\lVert z_i-\bar z\rVert_2^2,
\qquad \bar z=\frac{1}{n}\sum_{i=1}^{n}z_i.
$$

It requires at least two images per prompt. Diversity is averaged equally over prompts within each template and generation seed. The B2 summary reports per-seed/template metrics and their means and sample standard deviations across generation seeds.

FID here uses local torchvision ImageNet Inception-v3 features and preprocessing, which differ from standard TensorFlow/`pytorch-fid` FID. Its reference consists of generated SD 1.5 images, not real photographs. Report it as a measure of distributional change relative to that reference, with the feature extractor identified. DreamSim diversity should be read alongside alignment and composition scores: variation alone does not establish prompt correctness.

### T2I-CompBench prompt categories

CompBench uses validation prompt lists cached from source revision `4aa404212eb5d06e5adbcd9cee696c750d0d25a5`, with revision and SHA256 metadata verified by the loader. These are a separate benchmark corpus, not subdivisions of the 215 B2 prompts. The patterns below describe the semantic categories; the implementation reads complete prompt strings rather than generating them from these patterns.

| Category | Prompt structure being evaluated | Example from the cached validation list | Local composition scorer |
| --- | --- | --- | --- |
| Color | Multiple objects with specified colors | `a green bench and a blue bowl` | BLIP-VQA |
| Shape | Objects bound to specified shapes | `an oblong cucumber and a teardrop plum` | BLIP-VQA |
| Texture | Objects bound to specified textures or materials | `a plastic toy and a glass bottle` | BLIP-VQA |
| Spatial | Objects connected by a spatial relation | `a giraffe next to a lamp` | UniDet 2D spatial scorer |
| Non-spatial | Actions or interactions beyond positional relations | `A person is looking at a rainbow and marveling at its beauty.` | UniDet 2D spatial scorer in the current wrapper |
| Complex | Combined attributes and relations in a scene | `The red hat was on top of the brown coat rack.` | UniDet 2D spatial scorer in the current wrapper |

Each cached category contains **300 validation prompt entries**. The current runner uses the **first 50 per category (16.67%)**, for **300 selected category–prompt entries out of 1,800**. This is an ordered prefix, not a random sample; the B2 sampling fraction does not apply. The same selected prompts are reused across all baselines and generation seeds.

CompBench generates **10 images per prompt**: **500 images per category**, **3,000 images per baseline per generation seed**, and **9,000 image records per baseline across three generation seeds**. Inference uses 50 denoising steps and guidance scale 5.0. Replicate image seeds are `base_seed + replicate_index`; because the base seeds are 123, 124, and 126, their replicate-seed ranges overlap. Across-seed summaries should not be described as three disjoint sets of random seeds.

### T2I-CompBench evaluation metrics

**Composition scores:** the local wrapper invokes BLIP-VQA for color, shape, and texture, and UniDet's `2D_spatial_eval.py` for spatial, non-spatial, and complex categories. BLIP-VQA evaluates the requested object–attribute associations through visual questions. UniDet uses detected objects, detection confidence, and bounding-box relations to score positional consistency. Scores are saved per image and averaged within each category; higher scores indicate greater agreement according to the selected scorer.

The current mapping is an implementation-specific limitation: the same 2D spatial scorer is dispatched for non-spatial and complex prompts, without a `--complex` argument. Those outputs should not be presented as dedicated non-spatial or full complex-composition benchmark scores. The README documents the local dispatch rather than claiming equivalence to every category's intended upstream evaluation protocol.

**Distribution and diversity scores:** the pipeline also computes FID against paired `sd15` images and DreamSim within-prompt diversity for every category, using the B2 metric implementations described above. FID is computed per category and generation seed; DreamSim is computed over the ten replicates of each prompt and then averaged equally over prompts. Composition scores and distribution/diversity scores remain separate so that gains in one are not mistaken for gains in another.

### Recorded artifacts

Periodic evaluation computes CLIP and BERT rewards through `clover/utils/rewards_utils.py`. Each event records individual image scores, means, sample standard deviations, sample counts, and normal-approximation 95% confidence intervals. These intervals describe the 10 evaluated images within a run; they do not measure variation across training seeds or the full held-out prompt pool.

For standard launcher runs, configuration, history, and model artifacts are stored under `outputs/emo_v5/seed_<seed>/`; periodic evaluation metrics are appended to `evals/eval_metrics.json` within that directory. Evaluation images are saved under `evals/images/`. Training data and replay are stored under `clover/data/emo_v5/seed_<seed>/`; the current replay implementation uses `trajectories.pt` and clears it at buffer-reset boundaries.

Standalone B2 stage results and aggregate tables are saved under `clover/evaluate/metrics/<run_name>/`, including `b2_fid.json`, `b2_clip.json`, `b2_bert.json`, `b2_imagereward.json`, `b2_dreamsim.json`, and per-image/per-seed/summary tables. CompBench images and manifests are stored under `clover/data/<baseline>/t2i_compbench/<run_name>_seed_<seed>/`; metric files are stored under `outputs/<baseline>/evals/t2i_compbench/<run_name>_seed_<seed>/`, with `metrics.json` for composition and `fid.json`, `dreamsim.json`, and `distribution_summary.json` for distribution and diversity.

For comparisons, keep the prompt split, periodic evaluation subset, image-generation seed, sampling settings, and reward implementation fixed. Report results for each training seed, and distinguish the 65-prompt held-out pool from the 10-prompt subset actually scored during training.
