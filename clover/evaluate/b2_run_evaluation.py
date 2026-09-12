import gc
import json
import os
from pathlib import Path

import pandas as pd
import torch
from IPython.utils.capture import capture_output
from clover.evaluate.b2_evals import b2_evaluation as b2


def run_b2_evaluation(baselines, seeds, b2_run_name, images_per_prompt, fraction) -> None:
    """Run paired B2 evaluation and save JSON after each completed stage.

    Files: clover/evaluate/metrics/<b2_run_name>/b2_<evaluation>.json.
    Uses checkpoint seed 123 and cuda:0. Returns nothing and displays no results.
    Earlier JSON results remain available if a later evaluation fails.
    """
    def save_json(name, value):
        path = metrics_dir / f"{name}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    with capture_output():
        plan_path = b2.prepare_b2_run(
            baselines,
            run_name=b2_run_name,
            seeds=seeds,
            fraction=fraction,
            images_per_prompt=images_per_prompt,
            checkpoint_seed=123,
            output_root=b2.ROOT / "outputs",
            inference_kwargs=dict(
                height=512, width=512, num_inference_steps=50,
                guidance_scale=7.5, negative_prompt="blurry, low quality, distorted",
            ),
        )
        metrics_dir = b2.ROOT / "clover/evaluate/metrics" / b2_run_name
        metrics_dir.mkdir(parents=True, exist_ok=True)
        save_json("b2_plan", json.loads(plan_path.read_text()))
        gc.collect()
        torch.cuda.empty_cache()

        manifests = b2.generate_b2_images(plan_path, device="cuda:0", quiet=True)
        save_json("b2_image_manifests", manifests)

        fid = b2.score_b2_fid(
            plan_path, device="cuda:0", batch_size=16, per_prompt=False,
            inception_weights=Path(os.environ.get("TORCH_HOME", b2.ROOT / ".cache/torch"))
            / "hub/checkpoints/inception_v3_google-0cc3c7bd.pth",
        )
        save_json("b2_fid", fid)
        del fid

        clip = b2.score_b2_clip(plan_path, device="cuda:0", batch_size=16)
        save_json("b2_clip", clip)
        del clip

        captions = b2.caption_b2_images(plan_path, device="cuda:0", batch_size=8)
        save_json("b2_captions", captions)
        del captions

        bert = b2.score_b2_bert(plan_path, device="cuda:0", batch_size=8)
        save_json("b2_bert", bert)
        del bert

        imagereward = b2.score_b2_imagereward(plan_path, device="cuda:0")
        save_json("b2_imagereward", imagereward)
        del imagereward

        per_image, per_seed, summary = b2.summarize_b2_scores(plan_path)
        for name, frame in (("b2_per_image", per_image), ("b2_per_seed", per_seed),
                            ("b2_summary", summary)):
            # DataFrame serialization maps missing statistics (e.g. one-seed std) to null.
            save_json(name, json.loads(frame.to_json(orient="records")))
