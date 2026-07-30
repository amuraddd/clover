"""Test reward functions to diagnose why DDPO/DPOK show zero rewards."""

import torch
from PIL import Image
import numpy as np

# Test with a simple dummy image
dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
prompts = ["a test prompt"]

print("Testing reward functions...")
print("=" * 60)

# Test aesthetic reward
try:
    from clover.utils.rewards_utils import aesthetic_proxy_reward
    reward = aesthetic_proxy_reward([dummy_image], prompts, device="cpu")
    print(f"✓ Aesthetic reward: {reward.item():.4f}")
except Exception as e:
    print(f"✗ Aesthetic reward failed: {e}")

# Test CLIP reward
try:
    from clover.utils.rewards_utils import clip_reward
    print("\nLoading CLIP model...")
    reward = clip_reward([dummy_image], prompts, device="cpu")
    print(f"✓ CLIP reward: {reward.item():.4f}")
except Exception as e:
    print(f"✗ CLIP reward failed: {e}")
    import traceback
    traceback.print_exc()

# Test BERT reward
try:
    from clover.utils.rewards_utils import bert_reward
    print("\nLoading BERT model...")
    reward = bert_reward([dummy_image], prompts, device="cpu")
    print(f"✓ BERT reward: {reward.item():.4f}")
except Exception as e:
    print(f"✗ BERT reward failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete!")
