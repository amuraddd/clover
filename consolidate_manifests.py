"""Consolidate separate epoch manifest files into a single eval_manifest.json with epoch numbers."""

import json
from pathlib import Path

eval_dir = Path("outputs/ddpo/evals")

# Read all existing epoch manifest files
consolidated = []
for manifest_file in sorted(eval_dir.glob("epoch_*_manifest.json")):
    # Extract epoch number from filename
    epoch_str = manifest_file.stem.split("_")[1]  # e.g., "epoch_0001_manifest" -> "0001"
    epoch = int(epoch_str)
    
    with manifest_file.open("r") as f:
        entries = json.load(f)
    
    # Add epoch number to each entry
    for entry in entries:
        entry["epoch"] = epoch
        consolidated.append(entry)

# Save consolidated manifest
output_file = eval_dir / "eval_manifest.json"
with output_file.open("w") as f:
    json.dump(consolidated, f, indent=2, sort_keys=True)

print(f"Consolidated {len(consolidated)} entries from {len(list(eval_dir.glob('epoch_*_manifest.json')))} epoch manifest files")
print(f"Saved to: {output_file}")
