"""Periodic EMO-v5 FID against SD 1.5 and within-prompt DreamSim diversity."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import torch

from clover.evaluate.b2_evals.b2_evaluation import (
    ROOT, DREAMSIM_BACKEND, DREAMSIM_WORKER, setup_b2_dreamsim, _save,
)

SAMPLES_PER_PROMPT = 10
EVAL_SEEDS = (123, 456, 789)
REFERENCE_MODEL = 'runwayml/stable-diffusion-v1-5'


def evaluation_data_dir(config):
    """Keep sweep runs distinct even when they share a training seed."""
    run = hashlib.sha256(str(Path(config.output_dir).resolve()).encode()).hexdigest()[:16]
    return ROOT / 'clover/data/emo_v5/evals' / run


@torch.no_grad()
def sd15_reference_images(config, prompts, device, dtype, *, eval_seed=EVAL_SEEDS[0]):
    """Cache independent, unmodified SD 1.5 samples by generation settings."""
    from diffusers import StableDiffusionPipeline, DDIMScheduler

    metadata = dict(model_id=REFERENCE_MODEL, prompts=prompts,
                    samples_per_prompt=SAMPLES_PER_PROMPT, seed=eval_seed,
                    height=config.height, width=config.width,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    negative_prompt=config.negative_prompt,
                    scheduler='DDIMScheduler', eta=0.0, dtype=str(dtype),
                    device_type=device.type, safety_checker=False)
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:24]
    folder = ROOT / 'clover/data/emo_v5/evals/sd15_reference' / key
    paths = [folder / f'{i:04d}.png' for i in range(len(prompts) * SAMPLES_PER_PROMPT)]
    manifest = folder / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) == metadata and all(p.is_file() for p in paths):
        return paths, metadata
    folder.mkdir(parents=True, exist_ok=True)
    # A separate pipeline ensures neither learned variance nor LoRA enters the reference.
    reference = StableDiffusionPipeline.from_pretrained(
        REFERENCE_MODEL, torch_dtype=dtype, safety_checker=None,
        requires_safety_checker=False,
    )
    try:
        reference.scheduler = DDIMScheduler.from_config(reference.scheduler.config)
        reference.enable_vae_slicing()
        if device.type == 'cuda':
            reference.enable_model_cpu_offload(gpu_id=device.index or 0)
        else:
            reference.to(device)
        for index, prompt in enumerate(prompts):
            images = reference(
                [prompt] * SAMPLES_PER_PROMPT,
                negative_prompt=[config.negative_prompt] * SAMPLES_PER_PROMPT,
                height=config.height, width=config.width,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale, eta=0.0,
                generator=torch.Generator(device=device).manual_seed(eval_seed + index),
            ).images
            if len(images) != SAMPLES_PER_PROMPT:
                raise ValueError('SD 1.5 returned an unexpected image count')
            for sample, image in enumerate(images):
                image.save(paths[index * SAMPLES_PER_PROMPT + sample])
        _save(manifest, metadata)
    finally:
        del reference
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return paths, metadata


def dreamsim_diversity(records, folder, device):
    """Run the existing locked DreamSim worker on the training device only."""
    python = DREAMSIM_BACKEND / '.venv/bin/python'
    if not python.is_file():
        setup_b2_dreamsim()
    request, result = folder / 'dreamsim_request.json', folder / 'dreamsim_results.json'
    _save(request, dict(records=records, batch_size=4, download_root=str(ROOT / '.cache/DreamSim')))
    env = os.environ.copy()
    worker_device = 'cpu'
    if device.type == 'cuda':
        index = device.index or 0
        visible = env.get('CUDA_VISIBLE_DEVICES')
        env['CUDA_VISIBLE_DEVICES'] = visible.split(',')[index] if visible else str(index)
        worker_device = 'cuda:0'
    env.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
    env.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
    with (folder / 'dreamsim.log').open('w') as log:
        subprocess.run([str(python), str(DREAMSIM_WORKER), str(request), str(result),
                        '--device', worker_device], cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    scored = json.loads(result.read_text())
    if [r['id'] for r in scored] != [r['id'] for r in records]:
        raise ValueError('DreamSim output IDs do not match evaluation images')
    per_prompt = []
    for index in sorted({r['prompt_index'] for r in records}):
        group = [r for r in scored if r['prompt_index'] == index]
        values = [float(r['dreamsim_diversity']) for r in group]
        if (len(group) != SAMPLES_PER_PROMPT or not all(math.isfinite(v) for v in values)
                or len(set(values)) != 1
                or any(r['dreamsim_image_count'] != SAMPLES_PER_PROMPT for r in group)):
            raise ValueError('Invalid per-prompt DreamSim result')
        per_prompt.append(dict(prompt_index=index, prompt=group[0]['prompt'],
                               count=len(group), score=values[0]))
    return dict(mean=sum(r['score'] for r in per_prompt) / len(per_prompt),
                per_prompt=per_prompt, estimator='trace_sample_covariance',
                correction=1, model='ensemble', normalize_embeds=True,
                higher_is_better=True)


def evaluate_fid_and_diversity(config, prompts, image_paths, device, dtype, folder, *, eval_seed=EVAL_SEEDS[0]):
    """Pool prompts for 2048-d FID; never pool prompts for DreamSim diversity."""
    from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
    from pytorch_fid.inception import InceptionV3

    if not prompts or len(image_paths) != len(prompts) * SAMPLES_PER_PROMPT:
        raise ValueError('Evaluation requires exactly ten images per prompt')
    references, reference_metadata = sd15_reference_images(config, prompts, device, dtype, eval_seed=eval_seed)
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
    try:
        generated_stats = calculate_activation_statistics(
            [str(p) for p in image_paths], model, batch_size=4, dims=2048,
            device=device, num_workers=0)
        reference_stats = calculate_activation_statistics(
            [str(p) for p in references], model, batch_size=4, dims=2048,
            device=device, num_workers=0)
        fid = float(calculate_frechet_distance(*generated_stats, *reference_stats))
        if not math.isfinite(fid):
            raise ValueError('Non-finite FID score')
    finally:
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    records = [dict(baseline='emo_v5', id=str(i), seed=eval_seed,
                    template='periodic', prompt_index=i // SAMPLES_PER_PROMPT,
                    prompt=prompts[i // SAMPLES_PER_PROMPT], image_path=str(Path(path).resolve()))
               for i, path in enumerate(image_paths)]
    diversity = dreamsim_diversity(records, folder, device)
    return dict(fid=fid, fid_metadata=dict(
        dimensions=2048, generated_count=len(image_paths), reference_count=len(references),
        reference=reference_metadata, reference_image_paths=[str(p) for p in references],
        aggregation='all_prompts_pooled', lower_is_better=True,
        small_sample_warning='Only 10 images per prompt; FID is a noisy small-sample estimate.'),
        dreamsim_diversity=diversity['mean'], dreamsim_diversity_details=diversity)
