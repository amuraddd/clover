"""Command-line entry point for Clover B2 evaluation runs."""

import argparse
from collections.abc import Sequence

from dotenv import load_dotenv

from clover.evaluate.b2_run_evaluation import run_b2_evaluation


DEFAULT_SEEDS = [123, 124, 126]
DEFAULT_IMAGES_PER_PROMPT = 10
DEFAULT_FRACTION = 0.8


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
	"""Accept only single-directory B2 run names."""
	stripped = value.strip()
	if not stripped or stripped in {".", ".."} or "/" in stripped:
		raise argparse.ArgumentTypeError(
			"run name must be a single directory name without path separators"
		)
	return stripped


def build_parser() -> argparse.ArgumentParser:
	"""Create the CLI parser for B2 evaluation runs."""
	parser = argparse.ArgumentParser(
		description="Launch the Clover B2 evaluation workflow from the command line.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"--baselines",
		nargs="+",
		required=True,
		help="Baselines to evaluate. Include sd15 exactly once alongside the trained baselines.",
	)
	parser.add_argument(
		"--seeds",
		nargs="+",
		type=int,
		default=list(DEFAULT_SEEDS),
		help="Seed identifiers to include in the paired B2 evaluation.",
	)
	parser.add_argument(
		"--b2-run-name",
		required=True,
		type=run_name,
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
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	"""Parse CLI arguments and launch the B2 evaluation run."""
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

	run_b2_evaluation(
		baselines=args.baselines,
		seeds=args.seeds,
		b2_run_name=args.b2_run_name,
		images_per_prompt=args.images_per_prompt,
		fraction=args.fraction,
	)

	print("B2 evaluation completed", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

