"""Run the EMO v4 parameter grid with the remaining defaults from main.py."""

from itertools import product
from pathlib import Path
import os
import subprocess
import sys
import time

from main import EXPERIMENT_SEEDS, allocated_gpu_ids, build_default_argv, save_json


REWARD_SCALES = (2, 5, 10, 20)
KL_COEFFICIENTS = (0.01, 0.05, 0.1)
DIVERSITY_THRESHOLDS = (0.25, 0.50, 0.75)
MODULE = "clover.baselines.emo_v4"
SWEEP_DIR = Path("outputs/emo_v4/evals/parameter_sweep")


def sweep_jobs(gpu_ids: list[int]):
    """Yield independent configurations while inheriting the launcher defaults."""
    for seed, reward_scale, kl_coefficient, diversity_threshold in product(
        EXPERIMENT_SEEDS, REWARD_SCALES, KL_COEFFICIENTS, DIVERSITY_THRESHOLDS
    ):
        run_name = (
            f"reward_{reward_scale}_kl_{kl_coefficient:g}"
            f"_diversity_{diversity_threshold:g}"
        )
        output_dir = Path("outputs") / f"emo_v4_{run_name}" / f"seed_{seed}"
        argv = build_default_argv(MODULE, seed=seed, gpu_ids=gpu_ids)[1:]
        for flag, value in (
            ("--reward-scale", reward_scale),
            ("--kl-coefficient", kl_coefficient),
            ("--diversity-threshold", diversity_threshold),
            ("--output-dir", output_dir),
        ):
            if flag in argv:
                argv[argv.index(flag) + 1] = str(value)
            else:
                argv.extend([flag, str(value)])
        yield run_name, argv, {
            "baseline": "emo_v4",
            "seed": seed,
            "reward_scale": reward_scale,
            "kl_coefficient": kl_coefficient,
            "diversity_threshold": diversity_threshold,
            "output_dir": str(output_dir),
        }


def main() -> list[dict[str, object]]:
    """Run sequential subprocesses on at most two scheduler-visible GPUs."""
    gpu_ids = allocated_gpu_ids()
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    results = []
    for run_name, argv, record in sweep_jobs(list(range(len(gpu_ids)))):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", run_name, *argv]
        print(f"Starting {run_name} seed={record['seed']}: {' '.join(argv)}", flush=True)
        start = time.monotonic()
        record["returncode"] = subprocess.run(command, env=environment, check=False).returncode
        record["execution_time"] = time.monotonic() - start
        record["gpu_ids"] = list(gpu_ids)
        results.append(record)
        save_json(SWEEP_DIR / "execution_time.json", {"runs": results})
        print(f"Finished {run_name}: exit code {record['returncode']}", flush=True)
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        run_name = sys.argv[2]
        del sys.argv[1:3]
        from clover.baselines.emo_v4 import EMOV2V2Config, parse_config, train

        # Keep each configuration's replay and training data independent.
        train(
            parse_config(EMOV2V2Config, __doc__),
            baseline_name=f"emo_v4/trajectories/{run_name}",
        )
    else:
        sys.exit(int(any(run["returncode"] != 0 for run in main())))
