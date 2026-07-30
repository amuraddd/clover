# Clover Project Progress

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
