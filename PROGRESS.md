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
