---
description: Progress log for Clover baseline implementation and experiment support work
ms.date: 2026-08-03
---

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
