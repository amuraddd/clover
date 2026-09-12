"""T2I-CompBench setup, checkpoint image generation, and official metric evaluation."""

from concurrent.futures import ThreadPoolExecutor

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import subprocess
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
REVISION = "4aa404212eb5d06e5adbcd9cee696c750d0d25a5"
REPOSITORY = "https://github.com/Karine-Huang/T2I-CompBench"
RAW = f"https://raw.githubusercontent.com/Karine-Huang/T2I-CompBench/{REVISION}"
DATA_ROOT = ROOT / "clover/data/t2i_compbench"
CATEGORIES = ("color", "shape", "texture", "spatial", "non_spatial", "complex")
METRICS = {"color": "blip_vqa", "shape": "blip_vqa", "texture": "blip_vqa", "spatial": "unidet"}
BACKEND = ROOT / "clover/evaluate/compbench_backend"
logger = logging.getLogger(__name__)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_compbench_prompts(*, categories=CATEGORIES, data_root=DATA_ROOT, download=False):
    """Return category -> validation prompt lists from the pinned official source.

    Explicit download=True fetches missing files only; cached data is verified
    against its saved SHA256 on later reads. The source manifest records revision.
    """
    categories = tuple(categories)
    if not categories or len(set(categories)) != len(categories) or set(categories) - set(CATEGORIES):
        raise ValueError(f"Select distinct categories from {CATEGORIES}")
    folder = Path(data_root) / "prompts" / REVISION
    result = {}
    for category in categories:
        path = folder / f"{category}_val.txt"
        metadata = path.with_suffix(".source.json")
        if not path.is_file():
            if not download:
                raise FileNotFoundError(f"Missing {path}; call load_compbench_prompts(download=True)")
            url = f"{RAW}/examples/dataset/{path.name}"
            with urlopen(url, timeout=60) as response:
                content = response.read()
            folder.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            _write_json(metadata, {"url": url, "revision": REVISION,
                                   "sha256": hashlib.sha256(content).hexdigest()})
        if not metadata.is_file():
            raise ValueError(f"Missing source metadata for {path}")
        meta = json.loads(metadata.read_text())
        if meta["revision"] != REVISION or meta["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Prompt source verification failed: {path}")
        result[category] = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not result[category]:
            raise ValueError(f"Empty prompt file: {path}")
    return result


def generate_compbench_images(model, *, prompts_by_category=None, categories=tuple(METRICS),
                              images_per_prompt=10, base_seed=123, run_name="default",
                              data_root=None, inference_kwargs=None, reward_fn=None):
    """Generate benchmark samples with a LoadedBaseline; return a manifest path.

    Uses one image per call, with base_seed + replicate reused across prompts and
    baselines. Ten replicates is the official protocol. A subset/custom prompts
    is recorded as such. Existing run directories are refused to avoid mixing runs.
    reward_fn(images, prompts) can add the project's reward on the same images.
    """
    if not isinstance(images_per_prompt, int) or images_per_prompt < 1 or images_per_prompt > 1_000_000:
        raise ValueError("images_per_prompt must be in [1, 1000000]")
    for name in (model.baseline, run_name):
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("baseline and run_name must be single directory names")
    prompts = (load_compbench_prompts(categories=categories) if prompts_by_category is None
               else {key: list(value) for key, value in prompts_by_category.items()})
    if not prompts or set(prompts) - set(CATEGORIES):
        raise ValueError(f"Unknown/empty categories; choose from {CATEGORIES}")
    # Upstream extracts prompts from filenames and ids after the first underscore.
    for category, items in prompts.items():
        if not items or len(items) * images_per_prompt > 1_000_000:
            raise ValueError(f"Invalid sample count for {category}")
        for prompt in items:
            if (not isinstance(prompt, str) or not prompt.strip() or
                any(char in prompt for char in ('_', '/', '\\', '\n', '\r', '\x00')) or
                len(f"{prompt}_000000.png".encode()) > 255):
                raise ValueError(f"Prompt incompatible with official filename parser: {prompt!r}")
    root = Path(data_root) if data_root is not None else ROOT / "clover/data"
    run = (root / model.baseline / "t2i_compbench" / run_name).resolve()
    run.mkdir(parents=True, exist_ok=False)
    overrides = dict(inference_kwargs or {})
    settings = {key: overrides.get(key, getattr(model.config, key)) for key in
                ("height", "width", "num_inference_steps", "guidance_scale", "negative_prompt")}
    for key in ("eta", "likelihood_scale"):
        if hasattr(model.config, key):
            settings[key] = getattr(model.config, key)
    manifest = {"schema_version": 1, "benchmark": "T2I-CompBench", "revision": REVISION,
                "baseline": model.baseline, "run_name": run_name,
                "checkpoint": str(Path(model.checkpoint).resolve()),
                "prompt_source": "official_validation" if prompts_by_category is None else "provided_prompts",
                "images_per_prompt": images_per_prompt, "base_seed": base_seed,
                "inference": settings, "complete": False, "records": []}
    manifest_path = run / "manifest.json"
    _write_json(manifest_path, manifest)
    for category, items in prompts.items():
        samples = run / category / "samples"
        samples.mkdir(parents=True)
        for prompt_index, prompt in enumerate(items):
            for replicate in range(images_per_prompt):
                question_id = prompt_index * images_per_prompt + replicate
                seed = base_seed + replicate
                images = model.generate([prompt], seed=seed, **overrides)
                if len(images) != 1:
                    raise ValueError("Expected one generated image")
                image_path = samples / f"{prompt}_{question_id:06d}.png"
                images[0].save(image_path)
                record = {"category": category, "prompt_index": prompt_index,
                          "prompt": prompt, "replicate": replicate, "seed": seed,
                          "question_id": question_id, "image": str(image_path.relative_to(run))}
                if reward_fn is not None:
                    reward = float(reward_fn(images, [prompt])[0])
                    if not math.isfinite(reward):
                        raise ValueError("Project reward is not finite")
                    record["project_reward"] = reward
                manifest["records"].append(record)
            _write_json(manifest_path, manifest)
        logger.info("Generated %s: %d images", category, len(items) * images_per_prompt)
    manifest["complete"] = True
    _write_json(manifest_path, manifest)
    return manifest_path


def _load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if not manifest.get("complete") or not manifest.get("records"):
        raise ValueError("Generation is incomplete; refusing to score a partial run")
    return path, manifest


def _metric_spec(category, upstream, category_dir):
    if category not in METRICS:
        raise ValueError(f"{category} needs CLIPScore/3-in-1; BLIP-VQA and UniDet do not cover it")
    if METRICS[category] == "blip_vqa":
        return (upstream / "BLIPvqa_eval", ["BLIP_vqa.py", "--out_dir", str(category_dir)],
                category_dir / "annotation_blip/vqa_result.json")
    return (upstream / "UniDet_eval", ["2D_spatial_eval.py", "--outpath", str(category_dir)],
            category_dir / "labels/annotation_obj_detection_2d/vqa_result.json")


def run_compbench_metrics(manifest_path, *, upstream_dir=DATA_ROOT / "upstream",
                          categories=None, python_command=None, output_root=ROOT / "outputs"):
    """Run official scorers sequentially, then save per-image/category JSON metrics.

    python_command defaults to the dedicated uv project's interpreter. Call
    from this notebook; at most one visible GPU is exposed to each scorer.
    Release/offload the generation model first to leave memory for metric models.
    """
    path, manifest = _load_manifest(manifest_path)
    upstream = Path(upstream_dir).resolve()
    marker = upstream / "clover_source.json"
    if not marker.is_file() or json.loads(marker.read_text()).get("revision") != REVISION:
        raise FileNotFoundError("Pinned metric source missing; call setup_compbench() in this notebook")
    selected = list(categories) if categories is not None else list(dict.fromkeys(
        record["category"] for record in manifest["records"] if record["category"] in METRICS))
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select at least one distinct metric category")
    available = {r["category"] for r in manifest["records"]}
    if set(selected) - available:
        raise ValueError("Requested category is not in the image manifest")
    for category in selected:
        _metric_spec(category, upstream, path.parent / category)
    if python_command is None and not (BACKEND / ".venv/bin/python").is_file():
        raise FileNotFoundError("Metric runtime missing; call setup_compbench(install_runtime=True) in this notebook")
    if "spatial" in selected and not (upstream / "UniDet_eval/experts/expert_weights/Unified_learned_OCIM_RS200_6x+2x.pth").is_file():
        raise FileNotFoundError("UniDet RS200 weight missing; call setup_compbench(install_runtime=True) in this notebook")
    command = list(python_command) if python_command is not None else [
        "uv", "run", "--project", str(BACKEND), "--no-sync", "python"]
    if not command:
        raise ValueError("python_command must not be empty")
    env = os.environ.copy()
    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[0]
    else:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    for category in selected:
        folder = path.parent / category
        cwd, args, result_path = _metric_spec(category, upstream, folder)
        if result_path.exists():
            raise FileExistsError(f"Metric result already exists: {result_path}; use collect_compbench_metrics")
        records = [r for r in manifest["records"] if r["category"] == category]
        expected = {str((path.parent / r["image"]).resolve()) for r in records}
        actual = {str(p.resolve()) for p in (folder / "samples").iterdir()}
        if expected != actual:
            raise ValueError(f"Sample files do not match manifest for {category}")
        env["PYTHONPATH"] = os.pathsep.join([str(cwd), str(cwd / "BLIP")])
        with (folder / "metric.log").open("w") as log:
            try:
                subprocess.run(command + args, cwd=cwd, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"{category} scorer failed; see {folder / 'metric.log'}") from exc
    return collect_compbench_metrics(path, categories=selected, output_root=output_root)


def collect_compbench_metrics(manifest_path, *, categories=None, output_root=ROOT / "outputs"):
    """Validate official result IDs, join scores to images, and save structured results."""
    path, manifest = _load_manifest(manifest_path)
    selected = list(categories) if categories is not None else list(dict.fromkeys(
        r["category"] for r in manifest["records"] if r["category"] in METRICS))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select distinct metric categories")
    per_image, summaries = [], {}
    for category in selected:
        _, _, result_path = _metric_spec(category, Path(), path.parent / category)
        records = [r for r in manifest["records"] if r["category"] == category]
        if not records:
            raise ValueError(f"No images for {category}")
        raw = json.loads(result_path.read_text())
        scores = {}
        for row in raw:
            index, score = int(row["question_id"]), float(row["answer"])
            if index in scores or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Duplicate ID or invalid {category} score: {row}")
            scores[index] = score
        expected = {r["question_id"] for r in records}
        if set(scores) != expected or len(expected) != len(records):
            raise ValueError(f"Score IDs do not match image IDs for {category}")
        values = list(scores.values())
        summaries[category] = {"metric": METRICS[category], "mean": sum(values) / len(values),
                               "image_count": len(values), "prompt_count": len({r['prompt_index'] for r in records})}
        for record in records:
            per_image.append({**record, "metric": METRICS[category], "score": scores[record["question_id"]]})
        logger.info("%s %s: %.6f", category, METRICS[category], summaries[category]["mean"])
    result = {"benchmark": manifest["benchmark"], "revision": manifest["revision"],
              "baseline": manifest["baseline"], "checkpoint": manifest["checkpoint"],
              "manifest": str(path), "inference": manifest["inference"],
              "images_per_prompt": manifest["images_per_prompt"], "base_seed": manifest["base_seed"],
              "prompt_source": manifest["prompt_source"], "summary": summaries, "per_image": per_image,
              "unscored_categories": sorted({r["category"] for r in manifest["records"]} - set(selected))}
    output = Path(output_root) / manifest["baseline"] / "evals/t2i_compbench" / manifest["run_name"] / "metrics.json"
    _write_json(output, result)
    return result


def fetch(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        temporary = path.with_suffix(path.suffix + ".download")
        with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(path)


def prepare_source():
    root = DATA_ROOT / "upstream"
    url = f"https://api.github.com/repos/Karine-Huang/T2I-CompBench/git/trees/{REVISION}?recursive=1"
    with urlopen(url, timeout=60) as response:
        tree = json.load(response)
    entries = [entry for entry in tree['tree'] if entry['type'] == 'blob' and
               (entry['path'].startswith(('BLIPvqa_eval/', 'UniDet_eval/')) or
                entry['path'] in ('License.txt', 'NOTICE.md', 'Readme.md'))]
    def download(entry):
        target = root / entry['path']
        fetch(f"{RAW}/{entry['path']}", target)
        content = target.read_bytes()
        git_sha = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if git_sha != entry['sha']:
            raise ValueError(f"Upstream source checksum mismatch: {target}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, entries))
    _write_json(root / 'clover_source.json', {'revision': REVISION, 'files': {e['path']: e['sha'] for e in entries}})
    prompts = load_compbench_prompts(download=True)
    print('Official validation prompt counts:', {k: len(v) for k, v in prompts.items()})
    return root



def setup_compbench(*, install_runtime=False):
    """Fetch official sources; optionally install the project-local metric runtime."""
    root = prepare_source()
    if not install_runtime:
        return root
    env = os.environ.copy()
    # Build against Torch's CUDA 11.7 runtime; do not alter the main training env.
    env['MAX_JOBS'] = '2'
    env.setdefault('UV_PYTHON_INSTALL_DIR', str(DATA_ROOT / 'uv-python'))
    subprocess.run(['uv', 'sync', '--locked', '--project', str(BACKEND)], env=env, check=True)
    python = BACKEND / '.venv/bin/python'
    subprocess.run([str(python), '-c',
        'from torch.utils.cpp_extension import CUDA_HOME; '
        'assert CUDA_HOME, "Load a CUDA 11.7-compatible toolkit module before metric setup"'],
        env=env, check=True)
    env['FORCE_CUDA'] = '1'
    # Detectron2 imports torch during its build; install after uv sync.
    subprocess.run(['uv', 'pip', 'install', '--python', str(python), '--no-build-isolation',
                    'git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13'],
                   env=env, check=True)
    subprocess.run(['uv', 'pip', 'install', '--python', str(python),
                    'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl'],
                   env=env, check=True)
    fetch('https://huggingface.co/shikunl/prismer/resolve/main/expert_weights/Unified_learned_OCIM_RS200_6x%2B2x.pth',
          root / 'UniDet_eval/experts/expert_weights/Unified_learned_OCIM_RS200_6x+2x.pth')
    # BLIP weights/tokenizer are downloaded by upstream on the first BLIP run.
    subprocess.run([str(python), '-c',
                    'import torch, spacy, detectron2, timm; import detectron2._C; '
                    'spacy.load("en_core_web_sm"); print("Metric runtime imports passed")'],
                   env=env, check=True)
    # Import both upstream entry points without creating metric models.
    for folder, module in [('BLIPvqa_eval', 'BLIP_vqa.py'), ('UniDet_eval', '2D_spatial_eval.py')]:
        cwd = root / folder
        env['PYTHONPATH'] = os.pathsep.join([str(cwd), str(cwd / 'BLIP')])
        subprocess.run([str(python), '-c',
                        f'import runpy; runpy.run_path({module!r}, run_name="clover_import_check")'],
                       cwd=cwd, env=env, check=True)
    print('Metric runtime imports verified. BLIP weights/tokenizer download on first scoring run.')

    return root
