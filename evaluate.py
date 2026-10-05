"""Command-line entry point for sequential B2 and T2I-CompBench evaluation."""

import argparse
import gc
import json
from collections.abc import Sequence

from dotenv import load_dotenv
import torch

from clover.evaluate.b2_run_evaluation import run_b2_evaluation
from clover.evaluate.inference import load_baseline_model
from clover.evaluate.t21_comp_bench import (
	generate_compbench_images,
	run_compbench_metrics,
	setup_compbench,
)

BASELINES = ["ddpo", "b2diffurl", "emo_v5", "sd15"] 
DEFAULT_SEEDS = [123, 124, 126]
DEFAULT_IMAGES_PER_PROMPT = 20
DEFAULT_FRACTION = 0.8
RUN_NAME = "b2_eval_v2"


def positive_int(value: str) -> int:
	"""Parse a positive integer for CLI arguments."""
	parsed = int(value)
	if parsed < 1:
		raise argparse.ArgumentTypeError("value must be a positive integer")
	return parsed


def fraction_value(value: str) -> float:
	"""Parse an evaluation fraction constrained to the open interval (0, 1]."""
	parsed = float(value)
	if not 0 < parsed <= 1:
		raise argparse.ArgumentTypeError("fraction must satisfy 0 < fraction <= 1")
	return parsed


def run_name(value: str) -> str:
	"""Accept only single-directory evaluation run names."""
	stripped = value.strip()
	if not stripped or stripped in {".", ".."} or "/" in stripped:
		raise argparse.ArgumentTypeError(
			"run name must be a single directory name without path separators"
		)
	return stripped


def build_parser() -> argparse.ArgumentParser:
	"""Create the CLI parser for B2 and T2I-CompBench evaluation runs."""
	parser = argparse.ArgumentParser(
		description="Run Clover B2 evaluation followed by T2I-CompBench for the same baselines and seeds.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"--baselines",
		nargs="+",
		required=False,
		default=list(BASELINES),
		help="Baselines to evaluate. Include sd15 exactly once alongside the trained baselines.",
	)
	parser.add_argument(
		"--seeds",
		nargs="+",
		type=int,
		default=list(DEFAULT_SEEDS),
		help="Generation seeds for both benchmarks (checkpoint seed remains 123).",
	)
	parser.add_argument(
		"--b2-run-name",
		required=False,
		type=run_name,
		default=RUN_NAME,
		help="Directory name used under clover/evaluate/metrics and clover/data/b2_evaluation.",
	)
	parser.add_argument(
		"--images-per-prompt",
		type=positive_int,
		default=DEFAULT_IMAGES_PER_PROMPT,
		help="Number of generated images per sampled B2 prompt.",
	)
	parser.add_argument(
		"--fraction",
		type=fraction_value,
		default=DEFAULT_FRACTION,
		help="Fraction of each B2 prompt template to evaluate.",
	)
	parser.add_argument(
		"--rollout-chunk-size",
		type=positive_int,
		default=32,
		help="EMO v5 evaluation rollout chunk size; omitted uses the saved config (default 32).",
	)
	parser.add_argument(
		"--compbench-run-name",
		type=run_name,
		help="CompBench run prefix; defaults to the B2 run name. Each seed gets a _seed_<seed> suffix.",
	)
	parser.add_argument(
		"--compbench-images-per-prompt",
		type=positive_int,
		default=10,
		help="Images per CompBench prompt, following the notebook's ten-replicate protocol.",
	)
	parser.add_argument(
		"--compbench-install-runtime",
		action="store_true",
		help="Install the project-local CompBench metric runtime and UniDet weights during setup.",
	)
	return parser


def run_compbench_evaluation(baselines, seeds, compbench_run_name, images_per_prompt=10,
                             *, rollout_chunk_size=None, install_runtime=False) -> None:
	"""Follow the notebook protocol on each baseline and generation seed.

	Like B2, use checkpoint seed 123. Score color, shape, texture, and spatial
	validation prompts in full; the B2 fraction does not apply here. Images and
	manifests live in clover/data/<baseline>/t2i_compbench/<run>_seed_<seed>/;
	metric JSON lives in outputs/<baseline>/evals/t2i_compbench/<run>_seed_<seed>/.
	Existing CompBench runs are refused by the generation helper.
	"""
	setup_compbench(install_runtime=install_runtime)
	for baseline in baselines:
		manifests = []
		model = load_baseline_model(baseline, seed=123, device="cuda:0")
		try:
			for seed in seeds:
				print(f"Generating T2I-CompBench: {baseline}, seed {seed}", flush=True)
				manifests.append(generate_compbench_images(
					model,
					run_name=f"{compbench_run_name}_seed_{seed}",
					images_per_prompt=images_per_prompt,
					base_seed=seed,
					inference_kwargs={
						"num_inference_steps": 50,
						"guidance_scale": 5.0,
						"rollout_chunk_size": rollout_chunk_size,
					},
				))
		finally:
			# Release the pipeline (including EMO's variance head) before scorers run.
			del model
			gc.collect()
			torch.cuda.empty_cache()
		for manifest in manifests:
			print(f"Scoring T2I-CompBench: {manifest}", flush=True)
			results = run_compbench_metrics(manifest)
			print(json.dumps(results["summary"], indent=2), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
	"""Run B2, then T2I-CompBench using the same baselines and generation seeds."""
	load_dotenv(dotenv_path=".env")
	args = build_parser().parse_args(argv)

	print("=" * 80)
	print("Launching Clover B2 evaluation")
	print("=" * 80)
	print(f"Baselines: {', '.join(args.baselines)}", flush=True)
	print(f"Seeds: {', '.join(str(seed) for seed in args.seeds)}", flush=True)
	print(f"Run name: {args.b2_run_name}", flush=True)
	print(f"Images per prompt: {args.images_per_prompt}", flush=True)
	print(f"Fraction: {args.fraction}", flush=True)
	print(f"Rollout chunk size: {args.rollout_chunk_size or 'saved config'}", flush=True)

	# run_b2_evaluation(
	# 	baselines=args.baselines,
	# 	seeds=args.seeds,
	# 	b2_run_name=args.b2_run_name,
	# 	images_per_prompt=args.images_per_prompt,
	# 	fraction=args.fraction,
	# 	rollout_chunk_size=args.rollout_chunk_size,
	# )

	print("B2 evaluation completed", flush=True)
	print("Launching T2I-CompBench evaluation", flush=True)
	run_compbench_evaluation(
		baselines=args.baselines,
		seeds=args.seeds,
		compbench_run_name=args.compbench_run_name or args.b2_run_name,
		images_per_prompt=args.compbench_images_per_prompt,
		rollout_chunk_size=args.rollout_chunk_size,
		install_runtime=args.compbench_install_runtime,
	)
	print("T2I-CompBench evaluation completed", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

