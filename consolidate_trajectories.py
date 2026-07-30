"""Consolidate separate epoch trajectory files into a single trajectories.json file."""

import json
from pathlib import Path

baseline_name = "ddpo"
data_dir = Path("clover/data")
trajectory_dir = data_dir / baseline_name / "trajectories"
baseline_data_dir = data_dir / baseline_name

# Read all existing epoch trajectory files
consolidated = []
if trajectory_dir.exists():
    for trajectory_file in sorted(trajectory_dir.glob("epoch_*.json")):
        with trajectory_file.open("r") as f:
            trajectory_data = json.load(f)
        consolidated.append(trajectory_data)
        print(f"Loaded {trajectory_file.name}: epoch {trajectory_data.get('epoch')}")

# Save consolidated trajectories
output_file = baseline_data_dir / "trajectories.json"
output_file.parent.mkdir(parents=True, exist_ok=True)
with output_file.open("w") as f:
    json.dump(consolidated, f, indent=2, sort_keys=True)

print(f"\nConsolidated {len(consolidated)} trajectories from {len(list(trajectory_dir.glob('epoch_*.json')))} epoch files")
print(f"Saved to: {output_file}")
