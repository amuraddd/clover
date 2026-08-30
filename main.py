"""Unified entry point for all Clover baselines.

This runs every selected baseline and seed on the allocated GPUs.
Simply run: python main.py

Available baselines:
    - b2diffurl: B2-DiffuRL with backward-progressive, branch-based sampling
    - ddpo: Denoising Diffusion Policy Optimization
    - dpok: Diffusion Policy Optimization with online KL regularization
    - emo: Entropy-maximizing optimization with diversity replay
    - emo_v2: Second-generation entropy-maximizing optimization baseline
    - md3po_sac: MD3PO with a reward-scaled maximum-entropy actor update
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
    # ("dpok", "clover.baselines.dpok"),
    # ("b2diffurl", "clover.baselines.b2diffurl"),
    # ("md3po", "clover.baselines.md3po"),
    # ("md3po_sac", "clover.baselines.md3po_sac"),
    # ("emo", "clover.baselines.emo"),
    ("emo_v2", "clover.baselines.emo_v2"),
    # ("ddpo", "clover.baselines.ddpo"),
]

DEFAULT_GPU_IDS = ("0","1") #"1"
MAX_ALLOCATED_GPUS = 3
EXPERIMENT_SEEDS = [123] #123, 456, 789


def allocated_gpu_ids() -> tuple[str, ...]:
    """Return at most two scheduler-visible GPU identifiers."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = (
        tuple(device.strip() for device in visible_devices.split(",") if device.strip())
        if visible_devices
        else DEFAULT_GPU_IDS
    )
    if not gpu_ids:
        raise RuntimeError(
            "MD3PO-SAC requires at least one GPU; request a GPU in the Slurm "
            "allocation or expose a device through CUDA_VISIBLE_DEVICES."
        )
    return gpu_ids[:MAX_ALLOCATED_GPUS]

DEFAULT_BASELINE_ARGS = {
    "train_epochs": 50,
    "rollouts_per_epoch": 256,
    "learning_rate": 1e-4,
    "gpu_ids": [0],
    "save_every": 5,
    "num_inference_steps": 50,
    "minibatch_size": 32,
    "ppo_epochs": 2,
    "sac_epochs": 4,
    "reward_scale": 20.0,
    "importance_ratio_clip": 1.0,
    "lora_alpha": 16,
    "min_log_prob_std": 1e-4,
    "rollout_chunk_size": 32,
    "guidance_scale": 5.0,
    "adam_epsilon": 1e-8,
    "eta": 1.0,
    # "max_grad_norm": 1.0,
    "clip_range": 1e-4,
    # "target_kl": 0.13,
    "reward_type": "clip",
    # The frozen diffusion model runs in FP16 to reduce memory; the loader keeps
    # only trainable LoRA parameters in FP32 so Adam updates remain stable.
    "md3po_sac_mixed_precision": True,
}


def build_default_argv(
    script_name: str,
    seed: int | None = None,
    gpu_ids: list[int] | None = None,
) -> list[str]:
    """Build shared CLI arguments for one baseline and experimental seed."""
    if seed is None:
        seed = EXPERIMENT_SEEDS[0]
    baseline_name = script_name.rsplit(".", maxsplit=1)[-1]
    output_dir = f"outputs/{baseline_name}/seed_{seed}"
    argv = [script_name]
    argv.extend(["--seed", str(seed)])
    argv.extend(["--output-dir", output_dir])
    argv.extend(["--train-epochs", str(DEFAULT_BASELINE_ARGS["train_epochs"])])
    argv.extend(["--rollouts-per-epoch", str(DEFAULT_BASELINE_ARGS["rollouts_per_epoch"])])
    argv.extend(["--minibatch-size", str(DEFAULT_BASELINE_ARGS["minibatch_size"])])
    argv.extend(["--learning-rate", str(DEFAULT_BASELINE_ARGS["learning_rate"])])
    argv.extend(["--save-every", str(DEFAULT_BASELINE_ARGS["save_every"])])
    argv.extend(["--num-inference-steps", str(DEFAULT_BASELINE_ARGS["num_inference_steps"])])
    argv.extend(["--guidance-scale", str(DEFAULT_BASELINE_ARGS["guidance_scale"])])
    argv.extend(["--adam-epsilon", str(DEFAULT_BASELINE_ARGS["adam_epsilon"])])
    argv.extend(["--eta", str(DEFAULT_BASELINE_ARGS["eta"])])
    # argv.extend(["--max-grad-norm", str(DEFAULT_BASELINE_ARGS["max_grad_norm"])])
    if script_name.endswith(("md3po_sac", "emo", "emo_v2")):
        argv.extend(["--sac-epochs", str(DEFAULT_BASELINE_ARGS["sac_epochs"])])
        argv.extend(["--reward-scale", str(DEFAULT_BASELINE_ARGS["reward_scale"])])
        argv.extend([
            "--importance-ratio-clip",
            str(DEFAULT_BASELINE_ARGS["importance_ratio_clip"]),
        ])
    else:
        argv.extend(["--clip-range", str(DEFAULT_BASELINE_ARGS["clip_range"])])
        epoch_flag = "--dpok-epochs" if script_name.endswith("dpok") else "--ppo-epochs"
        argv.extend([epoch_flag, str(DEFAULT_BASELINE_ARGS["ppo_epochs"])])
    argv.extend(["--lora-alpha", str(DEFAULT_BASELINE_ARGS["lora_alpha"])])
    argv.extend(["--min-log-prob-std", str(DEFAULT_BASELINE_ARGS["min_log_prob_std"])])
    argv.extend(["--rollout-chunk-size", str(DEFAULT_BASELINE_ARGS["rollout_chunk_size"])])
    argv.extend(["--reward-type", str(DEFAULT_BASELINE_ARGS["reward_type"])])
    if gpu_ids is None:
        gpu_ids = DEFAULT_BASELINE_ARGS.get("gpu_ids")
    if gpu_ids is not None:
        argv.append("--gpu-ids")
        argv.extend(str(gpu_id) for gpu_id in gpu_ids)



    use_mixed_precision = (
        DEFAULT_BASELINE_ARGS["md3po_sac_mixed_precision"]
        if script_name.endswith("md3po_sac")
        else True
    )
    if not use_mixed_precision:
        argv.append("--no-mixed-precision")

    return argv


def main() -> list[dict[str, object]]:
    """Run enabled jobs on one or two allocated GPUs using one process per job."""
    print("=" * 80)
    print("Running Clover baseline/seed jobs on up to two GPUs")
    print("=" * 80)

    gpu_ids = allocated_gpu_ids()
    jobs = [
        (name, module, seed)
        for seed in EXPERIMENT_SEEDS
        for name, module in BASELINES
    ]
    results: list[dict[str, object]] = []
    visible_devices = ",".join(gpu_ids)
    logical_gpu_ids = list(range(len(gpu_ids)))
    for name, module, seed in jobs:
        baseline_argv = build_default_argv(
            module, seed=seed, gpu_ids=logical_gpu_ids
        )
        command = [sys.executable, "-m", module, *baseline_argv[1:]]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = visible_devices
        print(
            f"Starting {name.upper()} seed={seed} on GPUs {visible_devices}",
            flush=True,
        )
        print(f"Arguments: {' '.join(baseline_argv[1:])}", flush=True)
        start_time = time.time()
        returncode = subprocess.run(command, env=environment, check=False).returncode
        execution_time = time.time() - start_time
        results.append({
            "baseline": name,
            "seed": seed,
            "gpu_ids": list(gpu_ids),
            "output_dir": f"outputs/{name}/seed_{seed}",
            "execution_time": execution_time,
            "returncode": returncode,
        })
        status = (
            "completed successfully"
            if returncode == 0
            else f"failed with exit code {returncode}"
        )
        print(
            f"{'✓' if returncode == 0 else '✗'} {name.upper()} seed={seed} "
            f"{status} on GPUs {visible_devices} in {execution_time:.2f} seconds",
            flush=True,
        )

    print("\n" + "=" * 80)
    print("All baseline and seed runs completed")
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
