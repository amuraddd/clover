---
description: Progress log for Clover baseline implementation and experiment support work
ms.date: 2026-08-03
---

## 2026-08-12 — Added resumable MD3PO learning-rate scheduling

- Added a `StepLR` scheduler to MD3PO that reduces the learning rate by a factor of 10 after every 10 completed training epochs.
- Extended the shared checkpoint helpers with optional scheduler state persistence while preserving compatibility with baselines and older checkpoints that do not contain scheduler state.
- MD3PO now restores the scheduler together with its model and optimizer and logs the learning rate in each epoch's metrics.
- No GPU experiment was launched.


# Clover Project Progress

## 2026-08-03

### Task: Fixed unified baseline runner CLI mismatch for checkpoint cadence

**Completed:**
- Added `--save-every` to the shared parser in `clover/baselines/common.py`
- Aligned the common CLI with the arguments emitted by `main.py`
- Removed the immediate argparse failure that was stopping `python -m main` before baseline execution

**Notes:**
- The failure surfaced as `main.py: error: unrecognized arguments: --save-every 25`
- Baseline config dataclasses already define `save_every`, so the issue was isolated to shared argument parsing

### Task: Fixed MD3PO rollout image-type mismatch during diversity filtering

**Completed:**
- Updated `clover/baselines/md3po.py` to convert rollout comparison frames back into PIL images before CLIP image similarity
- Aligned the data type passed into `clip_image_cosine_similarity()` with that helper's `.convert("RGB")` expectation
- Removed the runtime failure that stopped MD3PO after epoch 4 with `'numpy.ndarray' object has no attribute 'convert'`

**Notes:**
- The bug was in the saved-rollout diversity filter, not in epoch counting or history writing
- `history.json` stopped at epoch 4 because training crashed before appending epoch 5

### Task: Preserved SSIM image format while keeping PIL inputs for CLIP similarity

**Completed:**
- Kept the original NumPy image arrays for `calculate_ssim()` in `clover/baselines/md3po.py`
- Added separate PIL image variables for `clip_image_cosine_similarity()`
- Split the two comparison paths so each metric receives the image type it expects

**Notes:**
- SSIM still operates on array data passed through OpenCV grayscale conversion
- CLIP image similarity now uses dedicated PIL wrappers without mutating the SSIM inputs

### Task: Added CLIP prompt-to-prompt similarity utility

**Completed:**
- Added `clip_prompt_cosine_similarity()` to `clover/utils/rewards_utils.py`
- Kept `clip_reward()` unchanged and isolated the new behavior to the prompt similarity helper
- Implemented prompt similarity with OpenCLIP text embeddings from `ViT-H-14` with `laion2b_s32b_b79k` weights
- Returned cosine similarity between the two prompt embeddings, which is equivalent to `1 - cosine_distance`

**Notes:**
- The function accepts two prompt strings plus an optional device override
- The return value is a scalar CPU tensor for easy downstream logging or comparison

### Task: Added shared default CLI arguments to the unified baseline runner

**Completed:**
- Updated `main.py` to define a shared `DEFAULT_BASELINE_ARGS` configuration
- Added `build_default_argv()` to convert those defaults into the CLI flags expected by each baseline
- Passed the generated arguments to every baseline invocation through `sys.argv`
- Printed the effective arguments before each baseline starts for easier traceability

**Notes:**
- The shared defaults currently cover `seed`, `train_epochs`, `rollouts_per_epoch`, `learning_rate`, and `gpu_ids`
- Mixed precision can be disabled centrally by setting `mixed_precision` to `False` in `DEFAULT_BASELINE_ARGS`
- File diagnostics for `main.py` passed after the change

## 2026-07-30

### Task: Added reward normalization to all baselines

**Completed:**
- Created `normalize_rewards()` function in `clover/baselines/common.py`:
  - Normalizes rewards to have mean 0 and variance 1
  - Handles edge case where all rewards are identical (std ≈ 0)
  - Uses small epsilon (1e-8) to prevent division by zero
- Updated all three baselines to apply reward normalization:
  - **DDPO**: Normalizes rewards before returning from `collect_rollouts()`
  - **DPOK**: Normalizes rewards before returning from `collect_rollouts()`
  - **B2-DiffuRL**: Normalizes rewards before branch selection in `collect_branch_rollouts()`

**Implementation Details:**
- Normalization formula: `(rewards - mean) / std` where mean and std are computed across all rewards in the trajectory
- If std < epsilon, only mean-centering is applied: `rewards - mean`
- Normalization happens after reward computation but before policy updates
- For B2-DiffuRL, normalization occurs before branch selection to ensure consistent scaling

**Benefits:**
- Stabilizes training by preventing reward scale from affecting learning rate effectiveness
- Makes different reward functions more comparable
- Reduces variance in policy gradient estimates
- Standard practice in RL to improve training stability

---

### Task: Added CLIP and BERT reward functions

**Completed:**
- Extracted CLIP score and BERT score functions from `clover/exp/clip_bert_rewards.ipynb`
- Adapted functions to match the reward function signature used in `clover/utils/rewards_utils.py`
- Added `clip_reward()` function:
  - Uses OpenCLIP (ViT-H-14, laion2b_s32b_b79k) for text-image alignment
  - Computes cosine similarity between image and text features
  - Takes PIL images directly instead of reading from directory
- Added `bert_reward()` function:
  - Uses BLIP for image captioning (Salesforce/blip-image-captioning-base)
  - Uses BERTScore with DeBERTa (microsoft/deberta-large-mnli) for semantic similarity
  - Returns F1 scores between generated captions and ground-truth prompts
  - Supports batching for efficient processing
- Registered both functions in REWARD_REGISTRY

**Function Signatures:**
Both functions match the standard reward signature:
```python
(images: list[Image.Image], prompts: list[str], device: torch.device | str | None = None) -> Tensor
```

**Available Reward Types:**
- `"aesthetic"`: Simple proxy based on brightness, saturation, and contrast
- `"clip"`: CLIP-based text-image alignment score
- `"bert"`: BERTScore-based caption-prompt alignment

**Notes:**
- CLIP reward requires `open_clip` package
- BERT reward requires `transformers` and `bert_score` packages
- Both functions automatically use CUDA if available
- BERT reward includes configurable parameters (caption model, BERT model, max tokens, batch size)

---

### Task: Added data persistence functionality to all baselines

**Completed:**
- Created helper functions in `clover/baselines/common.py` for structured data saving:
  - `save_trajectory_data()`: Saves RL trajectories in JSON format to `clover/data/{baseline_name}/trajectories/`
  - `save_evaluation_metrics()`: Saves evaluation metrics and images to `outputs/{baseline_name}/evals/`
  - `save_training_data()`: Saves complete training history to `clover/data/{baseline_name}/training_history.json`
  
- Updated all three baselines (DDPO, DPOK, B2-DiffuRL) to use the new save functions:
  - Trajectory data saved after each rollout collection
  - Evaluation metrics saved after each epoch with metrics and generated images
  - Complete training history saved at the end of training
  
- All baselines now use the reward function from `clover/utils/rewards_utils.py` via `make_reward_fn()`

**Data Structure:**
- Training trajectories: `clover/data/{baseline_name}/trajectories/epoch_XXXX.json`
- Training history: `clover/data/{baseline_name}/training_history.json`
- Evaluation metrics: `outputs/{baseline_name}/evals/eval_epoch_XXXX.json`
- Evaluation images: `outputs/{baseline_name}/evals/epoch_XXXX_XX.png`

**Notes:**
- Trajectory data includes prompts, timesteps, rewards, and reward statistics for each epoch
- Evaluation metrics are saved in a structured JSON format for easy comparison across baselines
- All save functions create necessary directories if they don't exist
- Data is saved in JSON format for easy analysis and potential retraining

**Next Steps:**
- Test baselines to ensure data is saved correctly
- Consider adding visualization scripts for comparing baseline performance

---

### Task: Consolidated evaluation manifests into single file with epoch numbers

**Completed:**
- Modified `save_image_grid_outputs()` in `clover/utils/baseline_utils.py`:
  - Added optional `epoch` parameter
  - When epoch is provided, appends entries to consolidated `eval_manifest.json`
  - Includes epoch number in each manifest entry
  - Maintains backward compatibility for cases without epoch parameter
  
- Updated `save_evaluation_metrics()` in `clover/baselines/common.py`:
  - Passes epoch parameter to `save_image_grid_outputs()`
  - Ensures evaluation images are added to consolidated manifest
  
- Created consolidated manifest from existing epoch-specific files:
  - Combined `epoch_0001_manifest.json` through `epoch_0004_manifest.json`
  - Generated single `outputs/ddpo/evals/eval_manifest.json` with all epochs
  - Each entry now includes `"epoch"` field for filtering and analysis

**Data Structure:**
Before:
```
outputs/ddpo/evals/
  ├── epoch_0001_manifest.json
  ├── epoch_0002_manifest.json
  ├── epoch_0003_manifest.json
  └── epoch_0004_manifest.json
```

After:
```
outputs/ddpo/evals/
  └── eval_manifest.json  # Contains all epochs with epoch numbers
```

**Manifest Entry Format:**
```json
{
  "epoch": 1,
  "image": "outputs/ddpo/evals/epoch_0001_00.png",
  "index": 0,
  "prompt": "an impressionist painting of clovers under warm sunlight"
}
```

**Benefits:**
- Single source of truth for all evaluation data
- Easy to filter by epoch for analysis
- Simpler to track evaluation history over time
- Reduces file clutter in eval directory
- Better for downstream analysis and visualization tools

---

### Task: Consolidated trajectory data into single file with epoch numbers

**Completed:**
- Modified `save_trajectory_data()` in `clover/baselines/common.py`:
  - Now appends trajectories to a single consolidated `trajectories.json` file
  - Stores file at `clover/data/{baseline_name}/trajectories.json` instead of separate epoch files
  - Each trajectory entry includes epoch number for filtering
  - Maintains all trajectory data: prompts, timesteps, rewards, and reward statistics
  
- Created consolidated trajectory file from existing epoch-specific files:
  - Combined `epoch_0001.json` through `epoch_0004.json` for DDPO baseline
  - Generated single `clover/data/ddpo/trajectories.json` with all epochs
  - Each trajectory includes `"epoch"` field for easy filtering

**Data Structure:**
Before:
```
clover/data/ddpo/trajectories/
  ├── epoch_0001.json
  ├── epoch_0002.json
  ├── epoch_0003.json
  └── epoch_0004.json
```

After:
```
clover/data/ddpo/
  └── trajectories.json  # Contains all epochs with epoch numbers
```

**Trajectory Entry Format:**
```json
{
  "epoch": 1,
  "prompts": ["an impressionist painting of clovers under warm sunlight"],
  "timesteps": [958, 925, 892, ...],
  "rewards": [[0.0, 0.0, ...]],
  "reward_stats": {
    "mean": 0.0,
    "std": 0.0,
    "min": 0.0,
    "max": 0.0
  }
}
```

**Benefits:**
- Single source of truth for all trajectory data
- Easy to filter trajectories by epoch for analysis
- Simplified data management and reduced file clutter
- More efficient for loading and analyzing training history
- Consistent with evaluation manifest structure
- Easier to track RL training progression across epochs

---

### Task: Fixed zero rewards bug in DDPO and DPOK baselines

**Issue Identified:**
- DDPO and DPOK were producing all zero rewards during training
- Investigation of experiment logs showed:
  - DDPO: reward_mean = 0.0, reward_std = 0.0, loss = 0.0
  - DPOK: reward_mean = 0.0, reward_std = 0.0, loss = 0.0
  - B2-DiffuRL: Non-zero rewards (reward_mean: 2.23-5.10) ✓ Working correctly

**Root Cause:**
The reward computation in `collect_rollouts()` had a logic error:
```python
if timestep == 0:  # BUG: This condition is never true!
    images = decode_latents(pipe, latents)
    rewards.append(reward_fn(images, prompts).detach().float().cpu())
else:
    rewards.append(zero_rewards.clone())
```

The diffusion scheduler timesteps actually end at `1`, not `0`:
- Timesteps: `[958, 925, 892, ..., 34, 1]` (from trajectory data)
- The condition `timestep == 0` was never satisfied
- Reward function was never called, only zero_rewards were appended

**Fix Applied:**
Changed the condition to check for the last timestep in the scheduler:
```python
if timestep_tensor == pipe.scheduler.timesteps[-1]:  # FIX: Check for last timestep
    images = decode_latents(pipe, latents)
    rewards.append(reward_fn(images, prompts).detach().float().cpu())
else:
    rewards.append(zero_rewards.clone())
```

**Files Modified:**
- `clover/baselines/ddpo.py`: Fixed reward computation in `collect_rollouts()`
- `clover/baselines/dpok.py`: Fixed reward computation in `collect_rollouts()`

**Expected Outcome:**
- DDPO and DPOK should now compute non-zero rewards using the configured reward function
- Training should show proper reward signals for policy optimization
- Trajectories will contain actual reward values instead of all zeros

**Next Steps:**
- Re-run DDPO and DPOK baselines to verify fix
- Compare reward distributions across all three baselines
- Ensure training metrics show non-zero values

## 2026-08-02 — Offline trajectory storage

- Updated `save_trajectory_data()` to save complete rollout tensors with `torch.save`.
- Each run appends to one `clover/data/{baseline_name}/trajectories.pt` dictionary instead of creating per-trajectory files.
- Top-level rollout numbers increase monotonically, while each entry records its epoch, so repeated epoch numbers from later runs never overwrite existing data.

## 2026-08-03 — Configurable evaluation cadence

- Added `evaluate_every` to baseline configs in `clover/baselines/ddpo.py`, `clover/baselines/dpok.py`, and `clover/baselines/b2diffurl.py`.
- Updated each training loop to call `evaluate(...)` only when `epoch % evaluate_every == 0`.
- Moved DPOK and B2-DiffuRL evaluation from end-of-training to in-loop cadence-based evaluation for consistent behavior across baselines.
- Entries contain `epoch`, `state`, `action`, `prompts`, `rewards`, `timesteps`, `old_log_probs`, and `images`; model tensors are stored on CPU and images are RGB uint8 BCHW tensors.
- Added validation for missing rollout fields and atomic file replacement; verified repeated-epoch retention and PIL image reconstruction from reloaded tensors.

## 2026-08-03 — Evaluation metrics with CLIP and BERT rewards

- Updated `evaluate()` in `clover/baselines/common.py` to compute `clip_reward` and `bert_reward` on generated evaluation images.
- Added an `evaluate_metrics` payload containing `epoch`, `prompts`, `image_paths`, `clip_reward`, and `bert_reward`.
- Persisted metrics by appending to `outputs/{baseline_name}/evals/eval_metrics.json` on each evaluation run.
- Updated DDPO, DPOK, and B2-DiffuRL training loops to pass `epoch` into `evaluate(...)` so saved evaluation entries are epoch-tagged.

## 2026-08-03 — Max-diversity rollout aggregation in notebook

- Updated the notebook helper in `clover/exp/max_diversity_generation.ipynb` to compare the reference rollout final state against each prompt-matched rollout final state using SSIM.
- Changed the helper to collect only rollouts below the diversity threshold and merge `state`, `action`, `old_log_probs`, `timesteps`, `prompts`, and `rewards` into one returned rollout dict.
- Kept the fallback behavior deterministic by returning the reference rollout fields when no qualifying rollout is found.
- Refined tensor concatenation to preserve the leading rollout axis and concatenate on the sample axis, so shapes such as `[1, 30, 4, 64, 64]` now merge into `[1, 60, 4, 64, 64]` when a second qualifying rollout is included.
- Ensured the returned combined rollout always exposes `state`, `action`, `prompts`, `rewards`, `timesteps`, `old_log_probs`, and `images`.
- Fixed a runtime error in the same helper by inferring the concatenation axis per tensor field, which allows `images` tensors to concatenate on their batch axis instead of incorrectly forcing the state sample axis.

## 2026-08-03 — MD3PO rollout-format bridge

- Updated `md3p_combined_rollouts()` in `clover/baselines/md3po.py` to accept both live training rollouts (`states` and `actions`) and saved trajectory entries (`state` and `action`).
- Normalized the helper output back into the live training rollout format so `ppo_update()` continues to receive `states`, `actions`, `old_log_probs`, `timesteps`, `prompts`, `rewards`, and `images`.
- Loaded saved MD3PO trajectories with CPU mapping and `weights_only=True` to match the existing trajectory persistence path.

## 2026-08-03 — Restored deleted MD3PO baseline file

- Recreated `clover/baselines/md3po.py` after accidental deletion.
- Preserved the latest rollout-combination logic and the live-vs-saved rollout format bridge used by `md3p_combined_rollouts()`.
- Verified that `main.py` can resolve `clover.baselines.md3po` again with no editor diagnostics.

## 2026-08-03 — CLIP image similarity utility

- Added `clip_image_cosine_similarity()` to `clover/utils/rewards_utils.py`.
- The utility preprocesses two PIL images with CLIP, compares their normalized
  image embeddings using cosine similarity, and returns the score as a Python float.

## 2026-08-04 — Updated baseline optimization and sampling defaults

- Updated B2-DiffuRL, DDPO, DPOK, and MD3PO defaults to use Adam beta1=0.9 and beta2=0.999, a minibatch size of 64, and 256 rollout samples per training iteration.
- Wired the configured Adam betas into each baseline AdamW optimizer.
- Verified all four modules compile and their configuration defaults instantiate with the requested values.

## 2026-08-04 — Parallelized baseline execution across two GPUs

- Updated `main.py` to launch baselines as isolated subprocesses in pairs, assigning one Slurm-allocated GPU to each process.
- Made GPU assignment respect `CUDA_VISIBLE_DEVICES` and fail clearly when fewer than two GPUs are allocated.
- Updated `run_experiments.sh` to request exactly two GPUs and verified pair scheduling with mocked processes without launching GPU work.

## 2026-08-04 — Added DPOK clip-range compatibility field

- Added an unused `clip_range` placeholder to `DPOKConfig` so the shared baseline CLI can pass `--clip-range` without raising a constructor error.
- Verified DPOK configuration parsing with the shared clip-range argument.

## 2026-08-04 — Batched VAE latent decoding

- Updated the shared `decode_latents()` helper to decode rollout latents in batches of four, reducing peak VAE memory usage for every baseline.
- Preserved float32 baseline execution and verified chunk sizes, image count, and output ordering with a mock VAE.

## 2026-08-04 — Sample-wise MD3PO rollout batch aggregation

- Updated `md3po_combined_rollouts()` to encode every prompt with CLIP and every complete state trajectory as a normalized flattened vector.
- Added aligned per-sample dot-product scoring for both state diversity and prompt similarity, retaining only saved samples that pass both configured thresholds.
- Combined accepted states, actions, log probabilities, rewards, images, and prompts strictly on batch dimension 0 while preserving the shared timestep schedule.
- Verified sample-wise filtering and batch growth with a mocked CLIP encoder; a two-sample reference plus one accepted saved sample produced states shaped `(3, 1, 1, 1, 1)`.

## 2026-08-04 — Memory-bounded MD3PO rollout encoding

- Changed MD3PO state trajectory encoding to process at most four rollout samples at a time before concatenating normalized CPU encodings.
- Added reusable `clip_prompt_embeddings()` and `load_clip_text_encoder()` helpers to `clover/utils/rewards_utils.py` and routed MD3PO prompt encoding through them with a batch size of four.
- Verified ten-sample reference and saved rollouts produce encoder calls of `(4, 4, 2)` for both states and prompts, while preserving the expected combined batch size.

## 2026-08-04 — MD3PO method documentation

- Added `clover/baselines/README_MD3PO.md` describing rollout collection, memory-bounded state and CLIP prompt encoding, per-sample diversity and prompt-similarity filtering, batch-dimension replay aggregation, PPO optimization, and output persistence.
- Added Mermaid diagrams for the full MD3PO epoch flow and a concrete combined-rollout example in which 32 reference samples plus 10 accepted replay samples produce a batch of 42.
- Documented tensor shape contracts, threshold directions, shared timestep handling, and the current training-loop defaults.

## 2026-08-05 — Fixed aggregate execution summary persistence

- Corrected the `save_json()` argument order in `main.py` so the aggregate baseline execution summary is written to `execution_time.json` instead of raising a post-training `TypeError`.

## 2026-08-05 — Limited MD3PO replay to the previous iteration

- Updated MD3PO replay selection to combine the current reference rollout with samples from only the most recently saved trajectory, ignoring older trajectories.
- Made the pre-combination `reference_rollout` explicit in the training loop and ensured that only this current-iteration rollout is persisted as trajectory data.

## 2026-08-05 — Documented MD3PO functions

- Added behavior, argument, return-value, and applicable exception documentation to every top-level and nested function in `clover/baselines/md3po.py`.

## 2026-08-06 — Memory-bounded B2DiffuRL rollout inference

- Added B2DiffuRL-only UNet inference chunking with a default rollout chunk size of 32.
- Applied chunking to both shared-prefix sampling and expanded branch sampling while preserving classifier-free-guidance embedding alignment and sample ordering.
- Left the shared prediction helper and all other baselines unchanged.

## 2026-08-06 — Resumable baseline checkpoints

- Added atomic, overwriting checkpoints for DDPO, DPOK, B2DiffuRL, and MD3PO every five completed epochs.
- Checkpoints persist LoRA weights, optimizer state, completed epoch, metric history, and random-number-generator states under `outputs/{baseline}/checkpoint/checkpoint.pt`.
- Each baseline now restores an available checkpoint at startup and resumes from the epoch following the last completed checkpoint while retaining the final `lora_final` export.

## 2026-08-06 — Stabilized diffusion policy optimization and MD3PO replay

- Replaced zero-filled per-step rewards with raw terminal trajectory rewards broadcast across trainable denoising transitions for DDPO, DPOK, MD3PO, and B2DiffuRL.
- Excluded the final near-deterministic diffusion transition from policy objectives while retaining it for terminal image generation and reward evaluation.
- Reduced the default learning rate to `3e-6`, policy update passes to one, gradient norm to `0.1`, and set PPO clipping to `0.1`.
- Added target-KL early stopping at `0.1`, standard nonnegative approximate-KL measurement, and separate raw-reward and normalized-advantage metrics.
- Changed MD3PO replay to compare every sample from the latest previous iteration against every current sample using CLIP prompt and image embeddings, retaining previous samples that pass both semantic-similarity and visual-diversity filters against at least one current sample.
- Added compatibility handling that skips replay for legacy trajectories whose timestep schema includes the removed deterministic transition.

## 2026-08-06 — Retained only the latest MD3PO trajectory

- Updated trajectory persistence with an opt-in latest-only mode that atomically replaces the saved trajectory collection with the current epoch.
- Enabled latest-only retention for MD3PO, ensuring each iteration compares against exactly the immediately preceding iteration without accumulating obsolete trajectories.
- Left trajectory accumulation behavior unchanged for the other baselines.

## 2026-08-06 — Removed redundant checkpoint image saves

- Removed checkpoint-time image copies from DDPO, DPOK, B2DiffuRL, and MD3PO.
- Kept the per-epoch images saved alongside training evaluation metrics under each baseline's `training_evals/images` directory.

## 2026-08-07 — Stabilized DDPM policy gradients and expanded learning diagnostics

- Replaced timestep-number filtering with a shared posterior-standard-deviation criterion across DDPO, DPOK, B2-DiffuRL, and MD3PO, leaving deterministic terminal transitions in image sampling while excluding them from log-probability objectives.
- Removed artificial minimum variance from `ddpm_mean_std()` and made deterministic log-probability evaluation fail explicitly instead of producing saturated PPO ratios.
- Removed the pre-exponentiation PPO log-ratio clamp; retained standard PPO ratio clipping and added explicit non-finite-ratio handling.
- Switched shared experiment defaults to CLIP reward and increased learning rate to `1e-5`, LoRA alpha to 16, update epochs to 2, and maximum gradient norm to 1.0 across all baselines; MD3PO replay behavior remains unchanged.
- Added per-timestep KL/clipping, unclamped log-ratio percentiles, pre-clip gradient norms, LoRA parameter-update norms, prompt-category rewards, fixed-prompt evaluation confidence intervals, and separate MD3PO current/replay reward summaries.
- Added focused regression coverage and verified 49 of 50 Stable Diffusion scheduler transitions remain trainable while the zero-variance terminal transition is excluded; all four tests and compilation checks pass.

## 2026-08-07 — Registered generated baseline CLI arguments

- Updated the shared configuration parser to register `--lora-alpha` and `--min-log-prob-std`, plus the baseline-appropriate `--ppo-epochs` or `--dpok-epochs` flag based on fields present in each configuration dataclass.
- Added optional argument-list injection to `parse_config()` so generated command lines can be tested without launching model training.
- Added an integration regression test that passes each complete `main.py` argument vector through the corresponding MD3PO, DDPO, DPOK, or B2-DiffuRL parser and verifies the new overrides.
- Verified all five focused tests, Python compilation, module help construction, and scoped diff checks pass; no GPU experiment was launched.

## 2026-08-07 — Shared chunked classifier-free-guidance inference

- Moved classifier-free-guidance UNet chunking from B2-DiffuRL into the shared baseline utilities while preserving unconditional/conditional embedding alignment, output order, and autograd graphs.
- Routed MD3PO, DDPO, DPOK, and B2-DiffuRL rollout collection through the shared chunked helper with a configurable default chunk size of 32.
- Applied the same chunking to shared PPO updates and to both the trainable-policy and frozen-reference UNet forwards in DPOK updates.
- Registered `--rollout-chunk-size` in the shared dataclass-aware parser and added it to the generated arguments in `main.py`.
- Added functional regression coverage verifying a five-sample input with chunk size two creates CFG UNet batches `(4, 4, 2)`, preserves output order and gradients, and parses for all baseline configurations; all six focused tests and compilation checks pass.
- Did not launch a GPU experiment.

## 2026-08-08 — Isolated fixed evaluation prompt sampling

- Changed `standard_eval_prompts()` to use a private deterministic `random.Random(123)` instance instead of reseeding Python’s global random generator.
- Preserved reproducible fixed evaluation prompt selection without restarting the random sequence used to sample training prompts after evaluation epochs.
- Added regression coverage verifying evaluation prompt selection is deterministic and leaves the global training RNG state unchanged.

## 2026-08-08 — Added Inception-based diversity metrics

- Added `clover/utils/diversity_score.py` with normalized singleton-FID rollout scores, Inception Score, and dataset-level FID functions for PIL and tensor image batches.
- Replaced MD3PO’s CLIP image-similarity replay criterion with prompt-filtered, min-max-normalized Inception-feature FID scores using current rollout images as ground truth.
- Registered SciPy directly for the FID covariance square root and added CPU-only regression tests using injected lightweight models.


## 2026-08-08 — Added epoch-specific rollout seeds

- Updated DDPO, DPOK, B2-DiffuRL, and MD3PO to recreate the rollout generator at the start of every training epoch with `config.seed + epoch`.
- This makes each epoch’s latent-noise stream distinct while retaining reproducibility for a given base seed and epoch number, including resumed runs.
- Added regression coverage ensuring every baseline applies the epoch-specific seed policy.

## 2026-08-08 — Verified cached Inception model on CPU

- Confirmed the project-local Inception-v3 checkpoint has the expected `0cc3c7bd` SHA-256 prefix.
- Ran CPU-only smoke tests for normalized rollout FID and Inception Score with CUDA hidden; both completed successfully without attempting a download.
- No GPU experiment was launched.


## 2026-08-09 — Added configurable PPO likelihood scaling and diverse MD3PO replay

- Added a shared positive `likelihood_scale` for diffusion transition log-probabilities and wired it consistently through rollout collection and policy updates for DDPO, MD3PO, B2-DiffuRL, and DPOK.
- Enabled DDPO likelihood scaling at `1000.0`; retained backward-compatible `1.0` defaults for the other baselines and registered the `--likelihood-scale` CLI override. PPO KL and clip-fraction metrics use the resulting scaled likelihood ratios.
- Reversed MD3PO replay selection to retain prompt-compatible samples whose normalized singleton FID is greater than the diversity threshold.
- Added regression coverage for scaling, validation, defaults, and the MD3PO predicate. All 13 focused CPU tests and project-Python compilation checks pass; no GPU experiment was launched.


## 2026-08-09 — Matched DDPO paper PPO clipping defaults

- Changed the PPO clip range from `0.1` to the DDPO reference implementation value `1e-4` for DDPO, MD3PO, B2-DiffuRL, and the DPOK CLI-compatibility field.
- Updated the shared generated experiment arguments so `sbatch run_experiments.sh` launches all baselines with the same `1e-4` clip range.
- Restored DDPO to the paper-compatible unscaled mean transition likelihood (`likelihood_scale=1.0`); retained current optimizer batching and global advantage normalization for the next diagnostic run.
- Verified all 13 focused CPU tests, generated argument parsing for every baseline, compilation, and scoped whitespace checks pass. No GPU experiment was launched.

## 2026-08-10 — Analyzed DDPO and MD3PO training histories

- Analyzed 21 DDPO epochs and 17 MD3PO epochs using reward trends, prompt-category rewards, PPO diagnostics, gradient norms, and parameter-update norms.
- Found no statistically persuasive continued reward improvement: DDPO’s reward slope is small relative to epoch noise, while MD3PO’s current-policy reward is essentially flat.
- Observed that MD3PO’s combined reward can be distorted by variable replay composition and should not be used alone as evidence of policy improvement.
- No GPU experiment was launched and no training code was changed.

## 2026-08-10 — Diagnosed likelihood-ratio and KL behavior

- Traced transition log-probabilities and the shared PPO update used by DDPO, MD3PO, and B2-DiffuRL, and reviewed the saved DDPO/MD3PO run configurations.
- Identified cancellation-prone float32 approximate-KL logging, very small mean-reduced log-ratios, a mismatched KL stopping threshold, and off-policy replay concerns specific to MD3PO.
- Recommended stable KL diagnostics, calibrated likelihood aggregation and clipping, explicit behavior-policy handling for replay, and a short diagnostic ablation before full retraining.
- No GPU experiment was launched and no training code was changed.

## 2026-08-10 — Stabilized PPO approximate-KL computation

- Replaced the cancellation-prone float32 `exp(log_ratio) - 1 - log_ratio` PPO KL estimator with float64 `expm1(log_ratio) - log_ratio` in the shared update used by DDPO, MD3PO, and B2-DiffuRL.
- Added regression coverage for positive and negative log-ratios of magnitude `1e-5` and verified the shared PPO path calls the stable estimator.
- All 11 focused CPU stability tests, compilation checks, and scoped whitespace checks pass. No GPU experiment was launched.

## 2026-08-10 — Reanalyzed post-fix DDPO and MD3PO KL

- Reanalyzed the restarted two-epoch DDPO and MD3PO histories after the stable approximate-KL change.
- Confirmed the estimator now produces nonnegative, internally consistent KL values; the remaining small DDPO KL reflects genuinely tiny policy ratios rather than float32 cancellation.
- Found that KL and clipping are concentrated almost entirely at timestep 21, while MD3PO replay introduces a rare large negative log-ratio tail and substantially higher timestep-21 KL.
- No GPU experiment was launched and no training code was changed.

## 2026-08-11 — Fixed checkpoint resume RNG restoration

- Changed the shared checkpoint loader used by DDPO, MD3PO, DPOK, and B2-DiffuRL to load checkpoint tensors on CPU before restoring model, optimizer, and RNG state.
- This preserves saved CUDA RNG states as CPU ByteTensors and prevents `torch.cuda.set_rng_state_all` from failing during resume.
- Retained the existing per-baseline atomic `outputs/{baseline}/checkpoint/checkpoint.pt` behavior, which resumes from the most recently saved checkpoint.
