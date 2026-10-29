"""Run the EMO v5 parameter grid with the remaining defaults from main.py."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import product
from pathlib import Path
import os
import subprocess
import sys
import time

from main import (
    DEFAULT_BASELINE_ARGS,
    EXPERIMENT_SEEDS,
    build_default_argv,
    save_json,
)


REWARD_SCALES = (5, 10, 20)
CROSS_ENTROPY_COEFFICIENTS = (0.01, 0.05, 0.1)
DIVERSITY_THRESHOLDS = (0.0, 0.25, 0.50, 0.75)
# Verified against outputs/emo_v5/evals/parameter_sweep/execution_time.json
# and 25-epoch histories on 2026-10-22. All values remain needed for seed 456.
# Keep partial runs (reward=10, cross_entropy=0.05, seed=123) in the queue.
COMPLETED_COMBINATIONS = {
    (123, reward_scale, coefficient, threshold)
    for reward_scale, coefficients in ((5, (0.01, 0.05, 0.1)), (10, (0.01,)))
    for coefficient, threshold in product(coefficients, DIVERSITY_THRESHOLDS)
}
# Fixed EMO v5 settings inherited from main.py, outside the parameter sweep.
# GPU IDs come from Slurm; legacy baseline-only options do not apply to EMO v5.
FIXED_BASELINE_ARGS = {
    name: DEFAULT_BASELINE_ARGS[name]
    for name in (
        "train_epochs",
        "rollouts_per_epoch",
        "learning_rate",
        "save_every",
        "num_inference_steps",
        "minibatch_size",
        "sac_epochs",
        "gamma",
        "entropy_scale",
        "buffer_reset",
        "lora_alpha",
        "lora_rank",
        "min_log_prob_std",
        "rollout_chunk_size",
        "guidance_scale",
        "adam_epsilon",
        "eta",
        "max_grad_norm",
        "clip_range",
        "reward_type",
    )
}
MODULE = "clover.baselines.emo_v5"
SWEEP_DIR = Path("outputs/emo_v5/evals/parameter_sweep")


def sweep_jobs(gpu_ids: list[int]):
    """Yield independent configurations while inheriting the launcher defaults."""
    for seed, reward_scale, cross_entropy_coefficient, diversity_threshold in product(
        EXPERIMENT_SEEDS, REWARD_SCALES, CROSS_ENTROPY_COEFFICIENTS, DIVERSITY_THRESHOLDS
    ):
        if (seed, reward_scale, cross_entropy_coefficient, diversity_threshold) in COMPLETED_COMBINATIONS:
            continue
        run_name = (
            f"reward_{reward_scale}_cross_entropy_{cross_entropy_coefficient:g}"
            f"_diversity_{diversity_threshold:g}"
        )
        output_dir = Path("outputs") / f"emo_v5_{run_name}" / f"seed_{seed}"
        argv = build_default_argv(MODULE, seed=seed, gpu_ids=gpu_ids)[1:]
        parameters = {
            **FIXED_BASELINE_ARGS,
            "reward_scale": reward_scale,
            "cross_entropy_coefficient": cross_entropy_coefficient,
            "diversity_threshold": diversity_threshold,
        }
        for name, value in {**parameters, "output_dir": output_dir}.items():
            flag = f"--{name.replace('_', '-')}"
            if value is None:
                # Omitting gamma lets EMO v5 derive it from num_inference_steps.
                if flag in argv:
                    index = argv.index(flag)
                    del argv[index:index + 2]
            elif flag in argv:
                argv[argv.index(flag) + 1] = str(value)
            else:
                argv.extend([flag, str(value)])
        yield run_name, argv, {
            "baseline": "emo_v5",
            "seed": seed,
            **parameters,
            "output_dir": str(output_dir),
        }


def sweep_gpu_ids() -> tuple[str, ...]:
    """Use up to three distinct GPUs exposed by the Slurm allocation."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = tuple(dict.fromkeys(device.strip() for device in visible.split(",")
                                 if device.strip() and device.strip() != "-1"))
    if not gpu_ids:
        raise RuntimeError("Request GPUs with sbatch run_emo_v5.sh; "
                           "CUDA_VISIBLE_DEVICES must identify the allocated GPUs.")
    return gpu_ids[:3]


def run_job(job, gpu_id: str) -> dict[str, object]:
    """Run one subprocess with exactly one visible GPU (local CUDA device 0)."""
    run_name, argv, record = job
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", run_name, *argv]
    print(f"Starting {run_name} seed={record['seed']} GPU={gpu_id}: {' '.join(argv)}", flush=True)
    start = time.monotonic()
    record["returncode"] = subprocess.run(command, env=environment, check=False).returncode
    record["execution_time"] = time.monotonic() - start
    record["gpu_ids"] = [gpu_id]
    if record["returncode"] == 0:
        Path(
            f"clover/data/emo_v5/trajectories/{run_name}/seed_{record['seed']}/trajectories.pt"
        ).unlink(missing_ok=True)
        for image_subdir in ("evals/images", "training_evals/images"):
            for image_path in (Path(record["output_dir"]) / image_subdir).glob("*.png"):
                image_path.unlink(missing_ok=True)
    print(f"Finished {run_name} seed={record['seed']} GPU={gpu_id}: "
          f"exit code {record['returncode']}", flush=True)
    return record


def main() -> list[dict[str, object]]:
    """Keep one experiment active per GPU, refilling each free slot immediately."""
    gpu_ids = sweep_gpu_ids()
    jobs = iter(sweep_jobs([0]))
    results = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        pending = {}

        def submit_next(gpu_id):
            job = next(jobs, None)
            if job is not None:
                pending[executor.submit(run_job, job, gpu_id)] = gpu_id

        for gpu_id in gpu_ids:
            submit_next(gpu_id)
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                gpu_id = pending.pop(future)
                results.append(future.result())
                # Only this coordinator writes the shared summary.
                save_json(SWEEP_DIR / "execution_time.json", {"runs": results})
                submit_next(gpu_id)
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        run_name = sys.argv[2]
        del sys.argv[1:3]
        from clover.baselines.emo_v5 import EMOV2V2Config, parse_config, train

        # Keep each configuration's replay and training data independent.
        train(
            parse_config(EMOV2V2Config, __doc__),
            baseline_name=f"emo_v5/trajectories/{run_name}",
        )
    else:
        sys.exit(int(any(run["returncode"] != 0 for run in main())))
