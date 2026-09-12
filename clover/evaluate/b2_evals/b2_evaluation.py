"""Paired B2 evaluation stages for a single-GPU notebook.

Every stage persists CPU artifacts; model lifetimes end before the next stage.
CLIP and caption-BERTScore follow clover.utils.rewards_utils definitions.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = "sd15"
TEMPLATES = {1: "object_behavior", 2: "object_attribute", 3: "positional_relationship"}
CLIP_MODEL = "ViT-H-14"
CLIP_PRETRAINED = "laion2b_s32b_b79k"
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
BERT_MODEL = "microsoft/deberta-large-mnli"


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _release():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def checkpoint_status(baselines, *, checkpoint_seed=123, output_root=ROOT / "outputs", run_dirs=None, checkpoint_files=None):
    """Read-only checkpoint availability, without loading models or allocating GPU memory."""
    from clover.evaluate.inference import _BASELINE_TYPES, _VARIANCE_MODULES
    rows = []
    for baseline in baselines:
        if baseline == REFERENCE:
            rows.append({"baseline": baseline, "available": True, "run_dir": None, "checkpoint_file": None, "missing": []})
            continue
        if baseline not in _BASELINE_TYPES:
            raise ValueError(f"Unknown baseline: {baseline}")
        folder = Path(output_root) / baseline
        run = Path((run_dirs or {}).get(baseline, folder / f"seed_{checkpoint_seed}"))
        if not run.is_dir() and baseline not in (run_dirs or {}):
            run = folder
        checkpoint = Path((checkpoint_files or {}).get(baseline, run / "checkpoint/checkpoint.pt"))
        required = [run / "config.json", checkpoint]
        if baseline in _VARIANCE_MODULES:
            required.append(checkpoint.parent / "variance_head.pt")
        rows.append({"baseline": baseline, "available": all(p.is_file() for p in required),
                     "run_dir": str(run.resolve()), "checkpoint_file": str(checkpoint.resolve()), "missing": [str(p) for p in required if not p.is_file()]})
    return rows


def sample_b2_prompts(seeds=(123, 124, 126), fraction=0.8, images_per_prompt=10):
    """Build the shared prompt/image manifest once, independent of baseline order."""
    from clover.utils.prompts import TEMPLATE_1_EVAL_PROMPTS, TEMPLATE_2_EVAL_PROMPTS, TEMPLATE_3_EVAL_PROMPTS
    if not 0 < fraction <= 1 or not isinstance(images_per_prompt, int) or images_per_prompt < 1:
        raise ValueError("Require 0 < fraction <= 1 and positive integer images_per_prompt")
    if not seeds or len(set(seeds)) != len(seeds) or not all(isinstance(s, int) and s >= 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    records = []
    for seed in seeds:
        for template, prompts in enumerate((TEMPLATE_1_EVAL_PROMPTS, TEMPLATE_2_EVAL_PROMPTS, TEMPLATE_3_EVAL_PROMPTS), 1):
            count = math.floor(len(prompts) * fraction)
            if count < 1:
                raise ValueError("Sampling fraction leaves an empty template")
            # Each template uses the requested seed and samples without replacement.
            chosen = random.Random(seed).sample(list(enumerate(prompts)), count)
            for prompt_index, prompt in chosen:
                for replicate in range(images_per_prompt):
                    key = f"{seed}:{template}:{prompt_index}:{replicate}"
                    image_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
                    records.append({"id": key, "seed": seed, "template": template,
                        "template_name": TEMPLATES[template], "prompt_index": prompt_index,
                        "prompt": prompt, "replicate": replicate, "image_seed": image_seed,
                        "image": f"seed_{seed}/template_{template}/prompt_{prompt_index:03d}/image_{replicate:02d}.png"})
    return records


def prepare_b2_run(baselines, *, run_name="b2_eval_v1", seeds=(123, 124, 126), fraction=0.8,
                   images_per_prompt=10, checkpoint_seed=123, output_root=ROOT / "outputs",
                   data_root=ROOT / "clover/data", run_dirs=None, checkpoint_files=None, inference_kwargs=None):
    """Persist an immutable paired plan; fail before generation if any checkpoint is missing."""
    baselines = list(baselines)
    if REFERENCE not in baselines or len(set(baselines)) != len(baselines):
        raise ValueError("Include sd15 exactly once and use distinct baseline names")
    if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("run_name must be a single directory name")
    statuses = checkpoint_status(baselines, checkpoint_seed=checkpoint_seed, output_root=output_root, run_dirs=run_dirs, checkpoint_files=checkpoint_files)
    missing = [row for row in statuses if not row["available"]]
    if missing:
        raise FileNotFoundError(f"Supply run_dirs or remove unavailable baselines explicitly: {missing}")
    sources = {}
    for row in statuses:
        baseline = row["baseline"]
        if baseline == REFERENCE:
            from clover.baselines.ddpo import DDPOConfig
            sources[baseline] = {"model_id": DDPOConfig().model_id, "run_dir": None, "checkpoint_file": None, "files": {}}
        else:
            folder = Path(row["run_dir"])
            checkpoint = Path(row["checkpoint_file"])
            files = [folder / "config.json", checkpoint]
            variance = checkpoint.parent / "variance_head.pt"
            if variance.is_file():
                files.append(variance)
            sources[baseline] = {"run_dir": str(folder), "checkpoint_file": str(checkpoint), "files": {str(p): _hash(p) for p in files}}
    settings = {"height": 512, "width": 512, "num_inference_steps": 50,
                "guidance_scale": 7.5, "negative_prompt": "blurry, low quality, distorted"}
    unknown = set(inference_kwargs or {}) - set(settings)
    if unknown:
        raise ValueError(f"Unsupported generation settings: {unknown}")
    settings.update(inference_kwargs or {})
    if settings["height"] % 8 or settings["width"] % 8 or min(settings["height"], settings["width"], settings["num_inference_steps"]) <= 0:
        raise ValueError("Dimensions must be positive multiples of 8; steps must be positive")
    plan = {"schema_version": 1, "run_name": run_name, "baselines": [REFERENCE] + [b for b in baselines if b != REFERENCE],
            "seeds": list(seeds), "fraction": fraction, "images_per_prompt": images_per_prompt,
            "checkpoint_seed": checkpoint_seed, "sources": sources, "inference": settings,
            "data_root": str(Path(data_root).resolve()), "output_root": str(Path(output_root).resolve()),
            "records": sample_b2_prompts(seeds, fraction, images_per_prompt)}
    path = Path(data_root) / "b2_evaluation" / run_name / "plan.json"
    if path.is_file() and json.loads(path.read_text()) != plan:
        raise ValueError("Existing run has different inputs/checkpoints; choose a new run_name")
    _save(path, plan)
    return path.resolve()


def _plan(path):
    return json.loads(Path(path).read_text())


def _data(plan, baseline):
    return Path(plan["data_root"]) / baseline / "b2_evaluation" / plan["run_name"]


def _output(plan, baseline):
    return Path(plan["output_root"]) / baseline / "evals/b2" / plan["run_name"]


def generate_b2_images(plan_path, *, device="cuda:0", quiet=True):
    """Load one baseline at a time. Reuse verified completed images on rerun."""
    import torch
    from tqdm.auto import tqdm
    from clover.evaluate.inference import load_baseline_model
    if str(device) not in ("cpu", "cuda", "cuda:0"):
        raise ValueError("This notebook workflow uses only cuda:0 (or cpu for tests)")
    plan = _plan(plan_path)
    for baseline in plan["baselines"]:
        for filename, digest in plan["sources"][baseline]["files"].items():
            if _hash(filename) != digest:
                raise ValueError(f"Checkpoint changed after planning: {filename}; create a new run")
        folder = _data(plan, baseline)
        manifest_file = folder / "images.json"
        manifest = json.loads(manifest_file.read_text()) if manifest_file.is_file() else {
            "plan_sha256": _hash(plan_path), "baseline": baseline, "images": {}, "complete": False}
        if manifest["plan_sha256"] != _hash(plan_path):
            raise ValueError(f"Image manifest belongs to a different plan: {manifest_file}")
        pending = []
        for row in plan["records"]:
            path = folder / row["image"]
            saved = manifest["images"].get(row["id"])
            if saved and path.is_file() and saved["sha256"] == _hash(path):
                continue
            pending.append(row)
        if not pending:
            manifest["complete"] = True
            _save(manifest_file, manifest)
            if not quiet:
                print(f"{baseline}: {len(plan['records'])} verified images already exist")
            continue
        manifest["complete"] = False
        _save(manifest_file, manifest)
        model = load_baseline_model(baseline, seed=plan["checkpoint_seed"],
            run_dir=plan["sources"][baseline]["run_dir"],
            checkpoint_file=plan["sources"][baseline]["checkpoint_file"], device=device)
        try:
            manifest["model_id"] = model.config.model_id
            manifest["scheduler"] = dict(model.pipe.scheduler.config)
            manifest["dtype"] = str(model.dtype)
            for completed, row in enumerate(tqdm(pending, desc=f"Generate {baseline}", disable=quiet), 1):
                with torch.inference_mode():
                    image = model.generate(row["prompt"], seed=row["image_seed"], **plan["inference"])[0]
                path = folder / row["image"]
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp.png")
                image.save(temporary)
                temporary.replace(path)
                manifest["images"][row["id"]] = {**row, "sha256": _hash(path)}
                if completed % 10 == 0:
                    _save(manifest_file, manifest)
            manifest["complete"] = True
            _save(manifest_file, manifest)
        finally:
            _save(manifest_file, manifest)
            del model
            _release()
    return {b: str(_data(plan, b) / "images.json") for b in plan["baselines"]}


def _records(plan_path):
    """Refuse incomplete or modified image sets before computing any metric."""
    plan = _plan(plan_path)
    rows = []
    for baseline in plan["baselines"]:
        folder = _data(plan, baseline)
        manifest = json.loads((folder / "images.json").read_text())
        if not manifest["complete"] or manifest["plan_sha256"] != _hash(plan_path):
            raise ValueError(f"Incomplete/mismatched images for {baseline}")
        if set(manifest["images"]) != {r["id"] for r in plan["records"]}:
            raise ValueError(f"Image IDs do not match the plan for {baseline}")
        for row in plan["records"]:
            path = folder / row["image"]
            if _hash(path) != manifest["images"][row["id"]]["sha256"]:
                raise ValueError(f"Image changed: {path}; regenerate before scoring")
            rows.append({**row, "baseline": baseline, "image_path": str(path)})
    return plan, rows


def _images(rows):
    from PIL import Image
    images = []
    for row in rows:
        with Image.open(row["image_path"]) as image:
            images.append(image.convert("RGB"))
    return images


def _image_digest(plan, baseline):
    manifest = json.loads((_data(plan, baseline) / "images.json").read_text())
    hashes = [(row["id"], manifest["images"][row["id"]]["sha256"]) for row in plan["records"]]
    return hashlib.sha256(json.dumps(hashes).encode()).hexdigest()


def _save_metric(plan_path, plan, rows, metric, metadata):
    for baseline in plan["baselines"]:
        own = [r for r in rows if r["baseline"] == baseline]
        _save(_output(plan, baseline) / f"{metric}.json", {
            "plan_sha256": _hash(plan_path), "images_sha256": _image_digest(plan, baseline),
            "metric": metric, "metadata": metadata, "records": own})


def _cached_metric(plan_path, plan, metric, metadata):
    records = []
    expected = {r["id"] for r in plan["records"]}
    for baseline in plan["baselines"]:
        path = _output(plan, baseline) / f"{metric}.json"
        if not path.is_file():
            return None
        saved = json.loads(path.read_text())
        if (saved["plan_sha256"] != _hash(plan_path) or saved.get("images_sha256") != _image_digest(plan, baseline) or saved["metadata"] != metadata or
            len(saved["records"]) != len(expected) or {r['id'] for r in saved['records']} != expected):
            raise ValueError(f"Stale metric cache: {path}; use a new run_name")
        records.extend(saved["records"])
    return records


def _load_local_inception(weights_path, device):
    """Load torchvision ImageNet Inception weights strictly from disk, without URLs."""
    import torch
    from torchvision.models import inception_v3
    network = inception_v3(weights=None, aux_logits=True, transform_input=False, init_weights=False)
    network.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
    network.fc = torch.nn.Identity()
    network.requires_grad_(False)
    return network.to(device).eval()


def score_b2_fid(plan_path, *, device="cuda:0", batch_size=16, per_prompt=False,
                 inception_weights=None):
    """FID with local torchvision ImageNet Inception features; no model downloads.

    These weights/preprocessing differ from standard TensorFlow/pytorch-fid FID.
    The weight checksum and extractor identity are included in cached results.
    """
    import os
    from clover.utils.diversity_score import _inception_outputs, frechet_inception_distance
    from tqdm.auto import tqdm
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    weights = Path(inception_weights) if inception_weights is not None else (
        Path(os.environ.get("TORCH_HOME", ROOT / ".cache/torch")) /
        "hub/checkpoints/inception_v3_google-0cc3c7bd.pth")
    if not weights.is_file():
        raise FileNotFoundError(f"Local Inception weights missing: {weights}; pass inception_weights explicitly")
    extractor = {"implementation": "torchvision-imagenet-inception-v3", "dimensions": 2048,
                 "weights_path": str(weights.resolve()), "weights_sha256": _hash(weights),
                 "preprocessing": "bilinear-299; ImageNet mean/std; transform_input=False"}
    plan, rows = _records(plan_path)
    network = None
    features = {}
    try:
        for baseline in plan["baselines"]:
            own = [r for r in rows if r["baseline"] == baseline]
            folder = _output(plan, baseline)
            cache, info = folder / "fid_features.npy", folder / "fid_features.json"
            metadata = {"plan_sha256": _hash(plan_path), "images_sha256": _image_digest(plan, baseline), "ids": [r['id'] for r in own],
                        **extractor}
            if cache.is_file() and info.is_file() and json.loads(info.read_text()) == metadata:
                features[baseline] = np.load(cache, allow_pickle=False)
                if features[baseline].shape != (len(own), 2048):
                    raise ValueError(f"Invalid feature cache: {cache}")
                continue
            if network is None:
                network = _load_local_inception(weights, device)
            chunks = []
            for start in tqdm(range(0, len(own), batch_size), desc=f"FID features {baseline}"):
                images = _images(own[start:start+batch_size])
                chunks.append(_inception_outputs(
                    images, device=device, batch_size=batch_size, features=True, model=network
                ).cpu().numpy())
            features[baseline] = np.concatenate(chunks)
            folder.mkdir(parents=True, exist_ok=True)
            np.save(cache, features[baseline], allow_pickle=False)
            _save(info, metadata)
    finally:
        del network
        _release()
    reference = features[REFERENCE]
    results = []
    for baseline in plan["baselines"]:
        for seed in plan["seeds"]:
            for template in TEMPLATES:
                indices = [i for i,r in enumerate(plan['records']) if r['seed']==seed and r['template']==template]
                fid = 0. if baseline == REFERENCE else frechet_inception_distance(
                    reference[indices], features[baseline][indices], precomputed_features=True)
                results.append({"baseline": baseline, "seed": seed, "template": template,
                    "template_name": TEMPLATES[template], "level": "template", "fid": fid,
                    "reference_baseline": REFERENCE, "image_count": len(indices)})
                if per_prompt:
                    for prompt_index in sorted({plan['records'][i]['prompt_index'] for i in indices}):
                        group = [i for i in indices if plan['records'][i]['prompt_index']==prompt_index]
                        fid = 0. if baseline == REFERENCE else frechet_inception_distance(
                            reference[group], features[baseline][group], precomputed_features=True)
                        results.append({"baseline": baseline, "seed": seed, "template": template,
                            "template_name": TEMPLATES[template], "level": "prompt", "prompt_index": prompt_index,
                            "fid": fid, "reference_baseline": REFERENCE, "image_count": len(group)})
    for baseline in plan['baselines']:
        _save(_output(plan, baseline)/'fid.json', {"plan_sha256": _hash(plan_path),
            'images_sha256': _image_digest(plan, baseline), 'reference_images_sha256': _image_digest(plan, REFERENCE),
            "metadata": {"reference": REFERENCE, **extractor,
                         "fid_function": "clover.utils.diversity_score.frechet_inception_distance",
                         "aggregation": "per seed/template; optional per prompt", "per_prompt": per_prompt},
            "records": [r for r in results if r['baseline']==baseline]})
    return results


def score_b2_clip(plan_path, *, device="cuda:0", batch_size=16):
    """Project CLIP reward: raw cosine similarity (not multiplied by 100)."""
    import torch
    from clover.utils.rewards_utils import load_clip_encoder
    from tqdm.auto import tqdm
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    plan, rows = _records(plan_path)
    metadata = {"model": CLIP_MODEL, "pretrained": CLIP_PRETRAINED, "score": "cosine_similarity"}
    cached = _cached_metric(plan_path, plan, 'clip', metadata)
    if cached is not None:
        return cached
    network, preprocess, tokenizer, target = load_clip_encoder(device)
    scored = []
    try:
        with torch.inference_mode():
            for start in tqdm(range(0,len(rows),batch_size), desc="CLIP"):
                batch = rows[start:start+batch_size]
                pixels = torch.stack([preprocess(im) for im in _images(batch)]).to(target)
                tokens = tokenizer([r['prompt'] for r in batch]).to(target)
                image_features, text_features = network.encode_image(pixels).float(), network.encode_text(tokens).float()
                image_features = torch.nn.functional.normalize(image_features,dim=-1)
                text_features = torch.nn.functional.normalize(text_features,dim=-1)
                values = (image_features*text_features).sum(-1).cpu().tolist()
                scored.extend({**r,"clip": float(v)} for r,v in zip(batch,values))
                del pixels, tokens, image_features, text_features
    finally:
        del network
        _release()
    _save_metric(plan_path, plan, scored, 'clip', metadata)
    return scored


def caption_b2_images(plan_path, *, device="cuda:0", batch_size=8):
    """Generate captions for BERTScore, then release the BLIP caption model."""
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from tqdm.auto import tqdm
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    plan, rows = _records(plan_path)
    metadata = {"model": CAPTION_MODEL, "max_new_tokens": 50, "do_sample": False}
    cached = _cached_metric(plan_path, plan, 'captions', metadata)
    if cached is not None:
        return cached
    processor = BlipProcessor.from_pretrained(CAPTION_MODEL)
    network = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL).to(device).eval()
    scored = []
    try:
        with torch.inference_mode():
            for start in tqdm(range(0,len(rows),batch_size), desc="BLIP captions"):
                batch = rows[start:start+batch_size]
                inputs = processor(images=_images(batch),return_tensors='pt').to(device)
                tokens = network.generate(**inputs,max_new_tokens=50,do_sample=False)
                captions = processor.batch_decode(tokens,skip_special_tokens=True)
                scored.extend({**r,'caption': text.strip()} for r,text in zip(batch,captions))
                del inputs,tokens
    finally:
        del network,processor
        _release()
    _save_metric(plan_path,plan,scored,'captions',metadata)
    return scored


def score_b2_bert(plan_path, *, device="cuda:0", batch_size=8):
    """BERTScore precision/recall/F1 of BLIP captions against the original prompts."""
    from bert_score import BERTScorer
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    plan, rows = _records(plan_path)
    metadata = {"model": BERT_MODEL, "caption_model": CAPTION_MODEL, "rescale_with_baseline": False}
    cached = _cached_metric(plan_path,plan,'bert',metadata)
    if cached is not None:
        return cached
    captions = _cached_metric(plan_path,plan,'captions',{"model": CAPTION_MODEL,"max_new_tokens":50,"do_sample":False})
    if captions is None:
        raise FileNotFoundError("Run caption_b2_images before score_b2_bert")
    scorer = BERTScorer(model_type=BERT_MODEL, lang='en', device=device,
                        batch_size=batch_size, rescale_with_baseline=False)
    scorer._tokenizer.model_max_length = min(scorer._tokenizer.model_max_length,512)
    try:
        precision,recall,f1 = scorer.score([r['caption'] for r in captions], [r['prompt'] for r in captions],batch_size=batch_size)
        scored=[{**r,'bert_precision':float(p),'bert_recall':float(q),'bert_f1':float(f)}
                for r,p,q,f in zip(captions,precision.cpu(),recall.cpu(),f1.cpu())]
    finally:
        del scorer
        _release()
    _save_metric(plan_path,plan,scored,'bert',metadata)
    return scored


def setup_b2_imagereward():
    """Install the separate ImageReward runtime using uv, without changing CLIP deps."""
    import os
    import subprocess
    backend = ROOT / "clover/evaluate/b2_imagereward_backend"
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(ROOT / ".cache/uv-b2"))
    temporary = ROOT / ".cache/uv-b2-tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(temporary)
    subprocess.run(["uv", "sync", "--locked", "--project", str(backend)], env=env, check=True)
    subprocess.run([str(backend / ".venv/bin/python"), "-c", "import ImageReward; print('ImageReward import passed')"], check=True)


def score_b2_imagereward(plan_path, *, device="cuda:0"):
    """Official ImageReward-v1.0 in its compatible uv runtime; one GPU, no job launch."""
    import os
    import subprocess
    plan,rows=_records(plan_path)
    metadata={"model":"ImageReward-v1.0","package":"image-reward==1.5"}
    cached=_cached_metric(plan_path,plan,'imagereward',metadata)
    if cached is not None:
        return cached
    backend = ROOT / "clover/evaluate/b2_imagereward_backend"
    python = backend / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError("Run setup_b2_imagereward() once in the notebook")
    if str(device) not in ("cpu", "cuda", "cuda:0"):
        raise ValueError("Use cuda:0 for the single-GPU ImageReward worker")
    folder = Path(plan_path).parent
    request,output = folder / "imagereward_request.json", folder / "imagereward_results.json"
    _save(request,{"download_root":str(ROOT/'.cache/ImageReward'),"records":rows})
    env=os.environ.copy()
    if device != 'cpu':
        env['CUDA_VISIBLE_DEVICES']=env.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    _release()
    log = folder / "imagereward.log"
    print(f"Scoring ImageReward; progress log: {log}")
    with log.open('w') as stream:
        subprocess.run([str(python),str(ROOT/'clover/evaluate/b2_imagereward_worker.py'),
                        str(request),str(output),'--device',str(device)],
                       env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    scored=json.loads(output.read_text())
    if [(r['baseline'],r['id']) for r in scored]!=[(r['baseline'],r['id']) for r in rows]:
        raise ValueError('ImageReward output IDs do not match request')
    _save_metric(plan_path,plan,scored,'imagereward',metadata)
    return scored


def summarize_b2_scores(plan_path):
    """Join every image score and report per-seed/template and across-seed metrics."""
    import pandas as pd
    plan,rows=_records(plan_path)
    image_records=[]; fid_records=[]
    for baseline in plan['baselines']:
        merged={r['id']:dict(r) for r in rows if r['baseline']==baseline}
        for metric,columns in [('clip',['clip']),('bert',['caption','bert_precision','bert_recall','bert_f1']),('imagereward',['imagereward'])]:
            payload=json.loads((_output(plan,baseline)/f'{metric}.json').read_text())
            if payload['plan_sha256']!=_hash(plan_path) or payload.get('images_sha256')!=_image_digest(plan,baseline) or {r['id'] for r in payload['records']}!=set(merged) or len(payload['records'])!=len(merged):
                raise ValueError(f'Incomplete/stale {metric} for {baseline}')
            for row in payload['records']:
                for key in columns:
                    value=row[key]
                    if key!='caption' and not math.isfinite(value):
                        raise ValueError(f'Nonfinite {metric} for {baseline}')
                    merged[row['id']][key]=value
        fid=json.loads((_output(plan,baseline)/'fid.json').read_text())
        if fid['plan_sha256']!=_hash(plan_path) or fid.get('images_sha256')!=_image_digest(plan,baseline) or fid.get('reference_images_sha256')!=_image_digest(plan,REFERENCE):
            raise ValueError('FID plan mismatch')
        fid_records.extend(r for r in fid['records'] if r['level']=='template')
        image_records.extend(merged.values())
    per_image=pd.DataFrame(image_records)
    summary=per_image.groupby(['baseline','seed','template','template_name'],sort=False).agg(
        image_count=('id','size'),clip=('clip','mean'),bert_f1=('bert_f1','mean'),imagereward=('imagereward','mean')).reset_index()
    fid_df=pd.DataFrame(fid_records)
    summary=summary.merge(fid_df[['baseline','seed','template','fid']],on=['baseline','seed','template'],validate='one_to_one')
    if len(summary)!=len(plan['baselines'])*len(plan['seeds'])*3:
        raise ValueError('Missing template-level metric rows')
    across_seeds=summary.groupby(['baseline','template','template_name'],sort=False)[['fid','clip','bert_f1','imagereward']].agg(['mean','std']).reset_index()
    across_seeds.columns=['_'.join(filter(None,c)) if isinstance(c,tuple) else c for c in across_seeds.columns]
    for baseline in plan['baselines']:
        folder=_output(plan,baseline)
        _save(folder/'per_image_scores.json',per_image[per_image.baseline==baseline].to_dict('records'))
        _save(folder/'summary.json',{'reference':REFERENCE,'seeds':plan['seeds'],'inference':plan['inference'],
            'fid_extractor':json.loads((folder/'fid.json').read_text()).get('metadata',{}),
            'per_seed_template':summary[summary.baseline==baseline].to_dict('records'),
            'across_seeds':json.loads(across_seeds[across_seeds.baseline==baseline].to_json(orient='records'))})
    return per_image,summary,across_seeds
