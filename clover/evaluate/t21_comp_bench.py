"""T2I-CompBench setup, checkpoint image generation, and official metric evaluation."""

from concurrent.futures import ThreadPoolExecutor

import hashlib
import json
import logging
import math
import os
import re
from pathlib import Path
import subprocess
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
REVISION = "4aa404212eb5d06e5adbcd9cee696c750d0d25a5"
REPOSITORY = "https://github.com/Karine-Huang/T2I-CompBench"
RAW = f"https://raw.githubusercontent.com/Karine-Huang/T2I-CompBench/{REVISION}"
DATA_ROOT = ROOT / "clover/data/t2i_compbench"
CATEGORIES = ("color", "shape", "texture", "spatial", "non_spatial", "complex")
METRICS = {
    "color": "blip_vqa", 
    "shape": "blip_vqa", 
    "texture": "blip_vqa", 
    "spatial": "unidet",
    "non_spatial": "unidet",
    "complex": "unidet"
}
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


def _planned_compbench_records(prompts, images_per_prompt, base_seed):
    return [{"category": category, "prompt_index": index, "prompt": prompt,
             "replicate": replicate, "seed": base_seed + replicate,
             "question_id": index * images_per_prompt + replicate,
             "image": f"{category}/samples/{prompt}_{index * images_per_prompt + replicate:06d}.png"}
            for category, items in prompts.items() for index, prompt in enumerate(items)
            for replicate in range(images_per_prompt)]


def find_compbench_images(baseline, *, run_name, base_seed, images_per_prompt,
                          inference_kwargs=None, data_root=None, prompts_by_category=None):
    """Return a validated complete run without loading a generation model.

    Missing or interrupted runs return None so generation can resume. Conflicting
    settings or damaged completed runs fail explicitly instead of reusing stale
    metrics. Runtime-only rollout_chunk_size does not change image identity.
    """
    for name in (baseline, run_name):
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("baseline and run_name must be single directory names")
    root = Path(data_root) if data_root is not None else ROOT / "clover/data"
    path = (root / baseline / "t2i_compbench" / run_name / "manifest.json").resolve()
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    expected = {"baseline": baseline, "run_name": run_name, "base_seed": base_seed,
                "images_per_prompt": images_per_prompt, "revision": REVISION}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing CompBench run has different settings: {path}; use a new run_name")
    for key, value in (inference_kwargs or {}).items():
        if key != "rollout_chunk_size" and manifest["inference"].get(key) != value:
            raise ValueError(f"Existing CompBench run has different {key}: {path}; use a new run_name")
    if not manifest.get("complete"):
        return None
    prompts = load_compbench_prompts(categories=tuple(METRICS)) if prompts_by_category is None else prompts_by_category
    expected_rows = _planned_compbench_records(prompts, images_per_prompt, base_seed)
    actual = [{key: row.get(key) for key in expected_rows[0]} for row in manifest["records"]]
    if actual != expected_rows:
        raise ValueError(f"Completed CompBench manifest does not match requested prompts: {path}")
    from PIL import Image
    for row in manifest["records"]:
        image_path = path.parent / row["image"]
        with Image.open(image_path) as image:
            image.verify()
    logger.info("Reusing completed CompBench images: %s", path)
    return path


def generate_compbench_images(model, *, prompts_by_category=None, categories=tuple(METRICS),
                              images_per_prompt=10, base_seed=123, run_name="default",
                              data_root=None, inference_kwargs=None, reward_fn=None, resume=False):
    """Generate benchmark samples with a LoadedBaseline; return a manifest path.

    Uses one image per call, with base_seed + replicate reused across prompts and
    baselines. Ten replicates is the official protocol. A subset/custom prompts
    is recorded as such. resume=True continues interrupted runs without replacing
    valid images; otherwise existing run directories are refused.
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
    overrides = dict(inference_kwargs or {})
    settings = {key: overrides.get(key, getattr(model.config, key)) for key in
                ("height", "width", "num_inference_steps", "guidance_scale", "negative_prompt")}
    for key in ("eta", "likelihood_scale"):
        if hasattr(model.config, key):
            settings[key] = getattr(model.config, key)
    manifest = {"schema_version": 1, "benchmark": "T2I-CompBench", "revision": REVISION,
                "baseline": model.baseline, "run_name": run_name,
                "checkpoint": str(Path(model.checkpoint).resolve()) if model.checkpoint is not None else None,
                "prompt_source": "official_validation" if prompts_by_category is None else "provided_prompts",
                "images_per_prompt": images_per_prompt, "base_seed": base_seed,
                "inference": settings, "complete": False, "records": []}
    manifest_path = run / "manifest.json"
    if run.exists():
        if not resume:
            raise FileExistsError(f"Run already exists: {run}")
        if not manifest_path.is_file():
            raise ValueError(f"Existing run has no manifest: {run}; use a new run_name")
        existing = json.loads(manifest_path.read_text())
        if any(existing.get(key) != value for key, value in manifest.items()
               if key not in ("complete", "records")):
            raise ValueError("Existing CompBench run has different generation settings; use a new run_name")
        if existing.get("prompts", prompts) != prompts:
            raise ValueError("Existing CompBench run has different prompts; use a new run_name")
        expected = _planned_compbench_records(prompts, images_per_prompt, base_seed)
        saved = existing["records"]
        if [{key: row.get(key) for key in expected[0]} for row in saved] != expected[:len(saved)]:
            raise ValueError("Existing CompBench records do not match requested prompts")
        if existing.get("complete"):
            return find_compbench_images(model.baseline, run_name=run_name, base_seed=base_seed,
                images_per_prompt=images_per_prompt, inference_kwargs=settings,
                data_root=root, prompts_by_category=prompts)
    run.mkdir(parents=True, exist_ok=True)
    manifest["prompts"] = prompts
    _write_json(manifest_path, manifest)
    for category, items in prompts.items():
        samples = run / category / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        for prompt_index, prompt in enumerate(items):
            for replicate in range(images_per_prompt):
                question_id = prompt_index * images_per_prompt + replicate
                seed = base_seed + replicate
                image_path = samples / f"{prompt}_{question_id:06d}.png"
                images = None
                if resume and image_path.is_file():
                    from PIL import Image
                    try:
                        with Image.open(image_path) as image:
                            image.verify()
                        with Image.open(image_path) as image:
                            images = [image.convert("RGB")]
                    except (OSError, SyntaxError):
                        pass  # interrupted PNG write: regenerate only this image
                if images is None:
                    images = model.generate([prompt], seed=seed, **overrides)
                    if len(images) != 1:
                        raise ValueError("Expected one generated image")
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
                          categories=None, python_command=None, output_root=ROOT / "outputs", resume=False):
    """Run official scorers sequentially, then save per-image/category JSON metrics.

    python_command defaults to the dedicated uv project's interpreter. Call
    from this notebook; at most one visible GPU is exposed to each scorer.
    Release/offload the generation model first to leave memory for metric models.
    resume=True validates and reuses existing category results, scoring only
    missing categories. Invalid cached results are rejected.
    """
    path, manifest = _load_manifest(manifest_path)
    upstream = Path(upstream_dir).resolve()
    selected = list(categories) if categories is not None else list(dict.fromkeys(
        record["category"] for record in manifest["records"] if record["category"] in METRICS))
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select at least one distinct metric category")
    available = {r["category"] for r in manifest["records"]}
    if set(selected) - available:
        raise ValueError("Requested category is not in the image manifest")
    for category in selected:
        _metric_spec(category, upstream, path.parent / category)
    pending = []
    for category in selected:
        _, _, result_path = _metric_spec(category, upstream, path.parent / category)
        if resume and result_path.is_file():
            records = [r for r in manifest["records"] if r["category"] == category]
            _read_compbench_scores(result_path, records, category)
            logger.info("Reusing official CompBench scores: %s", result_path)
        else:
            pending.append(category)
    if not pending:
        return collect_compbench_metrics(path, categories=selected, output_root=output_root)
    marker = upstream / "clover_source.json"
    if not marker.is_file() or json.loads(marker.read_text()).get("revision") != REVISION:
        raise FileNotFoundError("Pinned metric source missing; call setup_compbench()")
    if python_command is None and not (BACKEND / ".venv/bin/python").is_file():
        raise FileNotFoundError("Metric runtime missing; call setup_compbench(install_runtime=True) in this notebook")
    if "spatial" in pending and not (upstream / "UniDet_eval/experts/expert_weights/Unified_learned_OCIM_RS200_6x+2x.pth").is_file():
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
    for category in pending:
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


def _read_compbench_scores(result_path, records, category):
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
    return scores


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
        scores = _read_compbench_scores(result_path, records, category)
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



def _check_compbench_cuda(python, env):
    """Check the build toolkit against the backend Torch before native compilation."""
    result = subprocess.run(
        [str(python), '-c',
         'import json, torch; from torch.utils.cpp_extension import CUDA_HOME; '
         'print(json.dumps({"cuda_home": CUDA_HOME, "torch_cuda": torch.version.cuda}))'],
        env=env, check=True, capture_output=True, text=True,
    )
    runtime = json.loads(result.stdout)
    cuda_home, torch_cuda = runtime['cuda_home'], runtime['torch_cuda']
    if not cuda_home or not torch_cuda:
        raise RuntimeError("CompBench requires CUDA-enabled PyTorch and a CUDA toolkit with nvcc")
    nvcc = Path(cuda_home) / 'bin/nvcc'
    if not nvcc.is_file():
        raise RuntimeError(f"CompBench CUDA compiler is missing: {nvcc}")
    result = subprocess.run([str(nvcc), '--version'], env=env, check=True,
                            capture_output=True, text=True)
    release = re.search(r'release (\d+\.\d+)', result.stdout)
    if release is None:
        raise RuntimeError(f"Cannot determine the CUDA toolkit version from {nvcc}")
    toolkit_cuda = release.group(1)
    if toolkit_cuda != torch_cuda:
        raise RuntimeError(
            f"CompBench CUDA mismatch: {nvcc} uses {toolkit_cuda}, but backend PyTorch "
            f"uses {torch_cuda}. The backend dependencies must match the available toolkit."
        )
    print(f"CompBench build toolkit and PyTorch both use CUDA {toolkit_cuda}", flush=True)


def setup_compbench(*, install_runtime=False):
    """Fetch official sources; optionally install the project-local metric runtime."""
    root = prepare_source()
    if not install_runtime:
        return root
    env = os.environ.copy()
    # Match the node's CUDA 12.4 toolkit in the dedicated metric environment.
    env['MAX_JOBS'] = '2'
    env.setdefault('UV_PYTHON_INSTALL_DIR', str(DATA_ROOT / 'uv-python'))
    subprocess.run(['uv', 'sync', '--locked', '--project', str(BACKEND)], env=env, check=True)
    python = BACKEND / '.venv/bin/python'
    _check_compbench_cuda(python, env)
    env['FORCE_CUDA'] = '1'
    # Rebuild Detectron2 against this Torch; cached wheels may use the old CUDA ABI.
    subprocess.run(['uv', 'pip', 'install', '--python', str(python), '--no-build-isolation',
                    '--no-cache', '--reinstall-package', 'detectron2',
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


def prepare_compbench_distribution_plan(manifest_paths, *, output_root=ROOT / "outputs"):
    """Adapt paired CompBench manifests to the existing B2 FID/DreamSim contract.

    No images are copied. ``template`` denotes a CompBench category; ``seed``
    denotes the run's base seed, while ``image_seed`` retains each replicate's
    generation seed. Plans and image hashes are immutable to prevent stale reuse.
    """
    from clover.evaluate.b2_evals import b2_evaluation as b2

    loaded = [_load_manifest(path) for path in manifest_paths]
    baselines = [manifest["baseline"] for _, manifest in loaded]
    if b2.REFERENCE not in baselines or len(set(baselines)) != len(baselines):
        raise ValueError("Include sd15 exactly once and use distinct baseline names")
    reference_path, reference = next(item for item in loaded if item[1]["baseline"] == b2.REFERENCE)
    if reference["images_per_prompt"] < 2:
        raise ValueError("CompBench DreamSim requires at least two images per prompt")
    keys = ("category", "prompt_index", "prompt", "replicate", "seed", "question_id", "image")
    def canonical(manifest):
        rows = sorted(manifest["records"], key=lambda row: (row["category"], row["question_id"]))
        identities = [(r["category"], r["question_id"]) for r in rows]
        if len(set(identities)) != len(rows):
            raise ValueError("Duplicate CompBench image IDs")
        groups = {}
        for row in rows:
            groups.setdefault((row["category"], row["prompt_index"]), []).append(row)
        for group in groups.values():
            if (len(group) != manifest["images_per_prompt"] or
                    {r["replicate"] for r in group} != set(range(manifest["images_per_prompt"])) or
                    len({r["prompt"] for r in group}) != 1 or
                    any(r["seed"] != manifest["base_seed"] + r["replicate"] for r in group)):
                raise ValueError("Incomplete or inconsistent CompBench prompt replicates")
        return [{key: row[key] for key in keys} for row in rows]

    reference_rows = canonical(reference)
    records = [{**row, "id": f"{row['category']}:{row['question_id']}",
                "template": row["category"], "image_seed": row["seed"],
                "seed": reference["base_seed"], "image": str(Path("..") / row["image"])}
               for row in reference_rows]
    plan = {"schema_version": 1, "benchmark": "T2I-CompBench",
            "run_name": reference["run_name"], "baselines": baselines,
            "seeds": [reference["base_seed"]], "records": records,
            "templates": {r["category"]: r["category"] for r in records},
            "data_dirs": {}, "output_dirs": {}, "sources": {},
            "independent_metric_caches": True}
    image_manifests = {}
    for path, manifest in loaded:
        for key in ("revision", "run_name", "base_seed", "images_per_prompt"):
            if manifest[key] != reference[key]:
                raise ValueError(f"Unpaired CompBench manifests: {key}")
        if canonical(manifest) != reference_rows:
            raise ValueError("Unpaired CompBench prompts/images")
        baseline = manifest["baseline"]
        plan["data_dirs"][baseline] = str(path.parent / "distribution_metrics")
        plan["output_dirs"][baseline] = str((Path(output_root) / baseline /
            "evals/t2i_compbench" / manifest["run_name"]).resolve())
        plan["sources"][baseline] = {"manifest": str(path), "sha256": b2._hash(path)}
        hashes = {}
        for row in records:
            image = (Path(plan["data_dirs"][baseline]) / row["image"]).resolve()
            if not image.is_relative_to(path.parent):
                raise ValueError("CompBench image must be inside its run directory")
            hashes[row["id"]] = {"sha256": b2._hash(image)}
        image_manifests[baseline] = {"complete": True, "images": hashes}
    # Each baseline combination owns an immutable plan; per-baseline image and
    # metric identities stay stable when the same reference participates again.
    pair_key = hashlib.sha256(json.dumps(sorted(baselines)).encode()).hexdigest()[:16]
    plan_path = reference_path.parent / "distribution_metrics" / f"plan_{pair_key}.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing distribution plan differs; use a new run_name")
    # Check every existing image snapshot before updating any files.
    for baseline, saved in image_manifests.items():
        saved["plan_sha256"] = b2._metric_plan_hash(plan_path, plan, baseline)
        path = Path(plan["data_dirs"][baseline]) / "images.json"
        if path.exists() and json.loads(path.read_text()) != saved:
            raise ValueError("CompBench images changed; use a new run_name")
    _write_json(plan_path, plan)
    for baseline, saved in image_manifests.items():
        _write_json(Path(plan["data_dirs"][baseline]) / "images.json", saved)
    return plan_path


def run_compbench_distribution_metrics(manifest_paths, *, output_root=ROOT / "outputs",
                                       device="cuda:0", batch_size=8, inception_weights=None):
    """Score existing paired runs with B2's FID and DreamSim diversity definitions.

    FID uses local torchvision ImageNet Inception features against sd15, not
    standard TensorFlow FID. DreamSim is trace(sample covariance) of normalized
    embeddings within each prompt (higher means greater diversity). Reuses B2's
    locked local DreamSim runtime, validation, feature caches and score caches.
    Writes fid.json, dreamsim.json and distribution_summary.json beside metrics.json.
    """
    from clover.evaluate.b2_evals import b2_evaluation as b2

    if str(device) not in ("cpu", "cuda", "cuda:0"):
        raise ValueError("Use cuda:0 for the single-GPU metric pipeline")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    plan_path = prepare_compbench_distribution_plan(manifest_paths, output_root=output_root)
    fid = b2.score_b2_fid(plan_path, device=device, batch_size=batch_size,
                          inception_weights=inception_weights)
    dreamsim = b2.score_b2_dreamsim(plan_path, device=device, batch_size=batch_size)
    plan = json.loads(plan_path.read_text())
    summaries = {}
    for baseline in plan["baselines"]:
        summary = {}
        for category in plan["templates"]:
            rows = [r for r in dreamsim if r["baseline"] == baseline and r["category"] == category]
            prompts = {r["prompt_index"]: r["dreamsim_diversity"] for r in rows}
            fid_row = next(r for r in fid if r["baseline"] == baseline and r["template"] == category)
            summary[category] = {"fid": fid_row["fid"], "reference_baseline": b2.REFERENCE,
                                 "dreamsim_diversity": sum(prompts.values()) / len(prompts),
                                 "image_count": len(rows), "prompt_count": len(prompts)}
            logger.info("%s %s FID: %.6f; DreamSim diversity: %.6f", baseline, category,
                        summary[category]["fid"], summary[category]["dreamsim_diversity"])
        summaries[baseline] = summary
        _write_json(Path(plan["output_dirs"][baseline]) / "distribution_summary.json",
                    {"benchmark": "T2I-CompBench", "baseline": baseline,
                     "base_seed": plan["seeds"][0], "plan_sha256": b2._metric_plan_hash(plan_path, plan, baseline),
                     "grouping": "per base_seed/category/prompt; equal prompt mean for DreamSim",
                     "summary": summary})
    return summaries
