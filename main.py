"""Unified entry point for all Clover baselines.

This runs two baselines concurrently, with one baseline assigned to each GPU.
Simply run: python main.py

Available baselines:
    - b2diffurl: B2-DiffuRL with backward-progressive, branch-based sampling
    - ddpo: Denoising Diffusion Policy Optimization
    - dpok: Diffusion Policy Optimization with online KL regularization
"""

import os
import subprocess
import sys
import time
from dotenv import load_dotenv

env_path = ".env"
load_dotenv(dotenv_path=env_path)

from clover.utils.baseline_utils import save_json


BASELINES = [
    ("md3po", "clover.baselines.md3po"),
    ("ddpo", "clover.baselines.ddpo"),
    ("dpok", "clover.baselines.dpok"),
    ("b2diffurl", "clover.baselines.b2diffurl"),
]

DEFAULT_GPU_IDS = ("0", "1")


def allocated_gpu_ids() -> tuple[str, str]:
    """Return two GPU identifiers, respecting a scheduler-provided allocation."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = (
        tuple(device.strip() for device in visible_devices.split(",") if device.strip())
        if visible_devices
        else DEFAULT_GPU_IDS
    )
    if len(gpu_ids) < 2:
        raise RuntimeError(
            "Two GPUs are required; request two GPUs in the Slurm allocation "
            "or expose two devices through CUDA_VISIBLE_DEVICES."
        )
    return gpu_ids[0], gpu_ids[1]

DEFAULT_BASELINE_ARGS = {
    "seed": 123,
    "train_epochs": 10,
    "rollouts_per_epoch": 256,
    "learning_rate": 3e-6,
    "gpu_ids": [0, 1],
    "save_every": 5,
    "num_inference_steps": 50,
    "minibatch_size": 64,
    "guidance_scale": 5.0,
    "adam_epsilon": 1e-8,
    "eta": 1.0,
    "max_grad_norm": 0.1,
    "clip_range": 0.1,
    "target_kl": 0.1,
}


def build_default_argv(script_name: str) -> list[str]:
    """Build shared CLI arguments for baseline entry points."""
    argv = [script_name]
    argv.extend(["--seed", str(DEFAULT_BASELINE_ARGS["seed"])])
    argv.extend(["--train-epochs", str(DEFAULT_BASELINE_ARGS["train_epochs"])])
    argv.extend(["--rollouts-per-epoch", str(DEFAULT_BASELINE_ARGS["rollouts_per_epoch"])])
    argv.extend(["--learning-rate", str(DEFAULT_BASELINE_ARGS["learning_rate"])])
    argv.extend(["--save-every", str(DEFAULT_BASELINE_ARGS["save_every"])])
    argv.extend(["--num-inference-steps", str(DEFAULT_BASELINE_ARGS["num_inference_steps"])])
    argv.extend(["--guidance-scale", str(DEFAULT_BASELINE_ARGS["guidance_scale"])])
    argv.extend(["--adam-epsilon", str(DEFAULT_BASELINE_ARGS["adam_epsilon"])])
    argv.extend(["--eta", str(DEFAULT_BASELINE_ARGS["eta"])])
    argv.extend(["--max-grad-norm", str(DEFAULT_BASELINE_ARGS["max_grad_norm"])])
    argv.extend(["--clip-range", str(DEFAULT_BASELINE_ARGS["clip_range"])])
    argv.extend(["--target-kl", str(DEFAULT_BASELINE_ARGS["target_kl"])])
    argv.extend(["--minibatch-size", str(DEFAULT_BASELINE_ARGS["minibatch_size"])])

    gpu_ids = DEFAULT_BASELINE_ARGS.get("gpu_ids")
    if gpu_ids is not None:
        argv.append("--gpu-ids")
        argv.extend(str(gpu_id) for gpu_id in gpu_ids)

    

    if not DEFAULT_BASELINE_ARGS.get("mixed_precision", True):
        argv.append("--no-mixed-precision")

    return argv


def main() -> list[dict[str, object]]:
    """Run all baselines in pairs, assigning one process to each GPU."""
    print("=" * 80)
    print("Running Clover baselines two at a time across two GPUs")
    print("=" * 80)

    gpu_ids = allocated_gpu_ids()
    results: list[dict[str, object]] = []
    for batch_start in range(0, len(BASELINES), len(gpu_ids)):
        batch = BASELINES[batch_start : batch_start + len(gpu_ids)]
        processes = []

        for gpu_id, (name, module) in zip(gpu_ids, batch):
            baseline_argv = build_default_argv(module)
            command = [sys.executable, "-m", module, *baseline_argv[1:]]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"Starting {name.upper()} on GPU {gpu_id}", flush=True)
            print(f"Arguments: {' '.join(baseline_argv[1:])}", flush=True)
            processes.append(
                (name, gpu_id, time.time(), subprocess.Popen(command, env=environment))
            )

        # Wait for both jobs in this pair before assigning the next pair.
        for name, gpu_id, start_time, process in processes:
            returncode = process.wait()
            execution_time = time.time() - start_time
            results.append(
                {
                    "baseline": name,
                    "gpu_id": gpu_id,
                    "execution_time": execution_time,
                    "returncode": returncode,
                }
            )
            status = (
                "completed successfully"
                if returncode == 0
                else f"failed with exit code {returncode}"
            )
            print(
                f"{'✓' if returncode == 0 else '✗'} {name.upper()} {status} "
                f"on GPU {gpu_id} in {execution_time:.2f} seconds",
                flush=True,
            )

    print("\n" + "=" * 80)
    print("All baselines completed")
    print("=" * 80)
    return results


if __name__ == "__main__":
    start_time = time.time()
    baseline_results = main()
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    save_json(
        "execution_time.json",
        {"baseline": "total", "execution_time": total_time, "runs": baseline_results},
    )
