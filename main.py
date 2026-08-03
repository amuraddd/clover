"""Unified entry point for all Clover baselines.

This runs all baselines sequentially with default arguments.
Simply run: python main.py

Available baselines:
    - b2diffurl: B2-DiffuRL with backward-progressive, branch-based sampling
    - ddpo: Denoising Diffusion Policy Optimization
    - dpok: Diffusion Policy Optimization with online KL regularization
"""

import sys
import time
from dotenv import load_dotenv

env_path = ".env"
load_dotenv(dotenv_path=env_path)

# Import all baseline main functions
from clover.baselines.b2diffurl import main as b2diffurl_main
from clover.baselines.ddpo import main as ddpo_main
from clover.baselines.dpok import main as dpok_main
from clover.baselines.md3po import main as md3po_main

from clover.utils.baseline_utils import save_json


BASELINES = [
    ("md3po", md3po_main),
    ("ddpo", ddpo_main),
    ("dpok", dpok_main),
    ("b2diffurl", b2diffurl_main),
]

DEFAULT_BASELINE_ARGS = {
    "seed": 123,
    "train_epochs": 100,
    "rollouts_per_epoch": 1,
    "learning_rate": 10e-4,
    "gpu_ids": [0],
    "save_every": 25,
}


def build_default_argv(script_name: str) -> list[str]:
    """Build shared CLI arguments for baseline entry points."""
    argv = [script_name]
    argv.extend(["--seed", str(DEFAULT_BASELINE_ARGS["seed"])])
    argv.extend(["--train-epochs", str(DEFAULT_BASELINE_ARGS["train_epochs"])])
    argv.extend(["--rollouts-per-epoch", str(DEFAULT_BASELINE_ARGS["rollouts_per_epoch"])])
    argv.extend(["--learning-rate", str(DEFAULT_BASELINE_ARGS["learning_rate"])])
    argv.extend(["--save-every", str(DEFAULT_BASELINE_ARGS["save_every"])])

    gpu_ids = DEFAULT_BASELINE_ARGS.get("gpu_ids")
    if gpu_ids is not None:
        argv.append("--gpu-ids")
        argv.extend(str(gpu_id) for gpu_id in gpu_ids)

    if not DEFAULT_BASELINE_ARGS.get("mixed_precision", True):
        argv.append("--no-mixed-precision")

    return argv


def main():
    """Run all baseline experiments sequentially with shared default arguments."""
    print("=" * 80)
    print("Running all Clover baselines with shared default arguments")
    print("=" * 80)

    original_argv = sys.argv.copy()

    for i, (name, baseline_fn) in enumerate(BASELINES, 1):
        print(f"\n[{i}/{len(BASELINES)}] Starting {name.upper()} baseline...")
        print("-" * 80)

        try:
            sys.argv = build_default_argv(original_argv[0])
            print(f"Arguments: {' '.join(sys.argv[1:])}")
            start_time = time.time()
            baseline_fn()
            end_time = time.time()
            print(f"\n✓ {name.upper()} completed successfully in {end_time - start_time:.2f} seconds")
            save_json({"baseline": name, "execution_time": end_time - start_time}, f"execution_time.json")
        except Exception as e:
            print(f"\n✗ {name.upper()} failed with error: {e}")
            # Continue with other baselines even if one fails
            continue

        print("-" * 80)

    # Restore original argv
    sys.argv = original_argv

    print("\n" + "=" * 80)
    print("All baselines completed")
    print("=" * 80)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    save_json({"baseline": "total", "execution_time": total_time}, f"execution_time.json")
