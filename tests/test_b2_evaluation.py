"""
### B2 evaluation: paired seeds and vanilla SD 1.5 reference
`run_b2_evaluation(baselines, seeds, b2_run_name, images_per_prompt, fraction)` runs the complete single-GPU B2 workflow. It returns nothing and saves each completed stage immediately under `clover/evaluate/metrics/<b2_run_name>/`: `b2_fid.json`, `b2_clip.json`, `b2_captions.json`, `b2_bert.json`, and `b2_imagereward.json`, plus the plan, image manifest paths, and final summaries. Generation images and reusable feature caches retain their existing locations.
FID uses the local Inception weights and shared utility; BERTScore compares BLIP captions with prompts. The example retains your current `fraction=0.1`; use `0.8` for 80% of each template and choose a new run name when changing an existing plan. Results can be loaded separately for inspection after the function finishes.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
from clover.evaluate.b2_evals import b2_evaluation as b2


class B2EvaluationTests(unittest.TestCase):
    def test_prompt_sampling_counts_and_pairing(self):
        rows=b2.sample_b2_prompts()
        self.assertEqual(len(rows),1560)
        self.assertEqual(rows,b2.sample_b2_prompts())
        self.assertEqual(len({r['id'] for r in rows}),1560)
        for seed in [123,124,126]:
            for template,count in [(1,36),(2,8),(3,8)]:
                group=[r for r in rows if r['seed']==seed and r['template']==template]
                self.assertEqual(len(group),count*10)
                prompts={r['prompt_index'] for r in group}
                self.assertEqual(len(prompts),count)
                for prompt in prompts:
                    self.assertEqual({r['replicate'] for r in group if r['prompt_index']==prompt},set(range(10)))
        self.assertNotEqual([r['id'] for r in rows if r['seed']==123],[r['id'] for r in rows if r['seed']==124])

    def make_plan(self, root):
        folder=root/'outputs/ddpo/seed_123'
        (folder/'checkpoint').mkdir(parents=True)
        (folder/'config.json').write_text('{}')
        (folder/'checkpoint/checkpoint.pt').write_bytes(b'test weights')
        return b2.prepare_b2_run(['ddpo','sd15'],run_name='test',seeds=[123],fraction=.1,
            images_per_prompt=2,data_root=root/'data',output_root=root/'outputs',
            inference_kwargs={'height':8,'width':8,'num_inference_steps':2})

    def fake_loader(self, calls):
        def load(baseline,**kwargs):
            import torch
            def generate(prompt,**settings):
                calls.append((baseline,prompt,settings['seed']))
                return [Image.new('RGB',(8,8))]
            return SimpleNamespace(config=SimpleNamespace(model_id='test'),
                pipe=SimpleNamespace(scheduler=SimpleNamespace(config={})),dtype=torch.float32,generate=generate)
        return load

    def test_generation_resumes_and_preserves_paired_seeds(self):
        with TemporaryDirectory() as tmp:
            path=self.make_plan(Path(tmp)); calls=[]
            with patch('clover.evaluate.inference.load_baseline_model',side_effect=self.fake_loader(calls)) as loader:
                b2.generate_b2_images(path,device='cpu')
                self.assertEqual(loader.call_count,2)
                self.assertEqual([c[1:] for c in calls if c[0]=='sd15'],[c[1:] for c in calls if c[0]=='ddpo'])
                b2.generate_b2_images(path,device='cpu')
                self.assertEqual(loader.call_count,2)
                plan,rows=b2._records(path)
                Path(rows[0]['image_path']).write_bytes(b'corrupted')
                with self.assertRaises(ValueError): b2._records(path)
                previous=len(calls)
                b2.generate_b2_images(path,device='cpu')
                self.assertEqual(len(calls),previous+1)
                b2._records(path)
            # Changed checkpoint cannot silently reuse cached images.
            cp=next(p for p in plan['sources']['ddpo']['files'] if p.endswith('.pt'))
            Path(cp).write_bytes(b'new weights')
            with self.assertRaises(ValueError): b2.generate_b2_images(path,device='cpu')

    def test_plan_rejects_changed_inputs_and_missing_checkpoint(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); self.make_plan(root)
            with self.assertRaises(ValueError):
                b2.prepare_b2_run(['sd15','ddpo'],run_name='test',seeds=[124],fraction=.1,
                    images_per_prompt=2,data_root=root/'data',output_root=root/'outputs')
            with self.assertRaises(FileNotFoundError):
                b2.prepare_b2_run(['sd15','dpok'],data_root=root/'data',output_root=root/'outputs')

    def test_score_summary_joins_by_id_not_row_order(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); path=self.make_plan(root)
            with patch('clover.evaluate.inference.load_baseline_model',side_effect=self.fake_loader([])):
                b2.generate_b2_images(path,device='cpu')
            plan,rows=b2._records(path)
            for metric,fields in [('clip',{'clip':.3}),('bert',{'caption':'test','bert_precision':.5,'bert_recall':.6,'bert_f1':.55}),('imagereward',{'imagereward':1.2})]:
                b2._save_metric(path,plan,[{**r,**fields} for r in reversed(rows)],metric,{})
            for baseline in plan['baselines']:
                b2._save(b2._output(plan,baseline)/'fid.json',{'plan_sha256':b2._hash(path),'images_sha256':b2._image_digest(plan,baseline),
                    'reference_images_sha256':b2._image_digest(plan,'sd15'),'records':[
                    {'baseline':baseline,'seed':123,'template':template,'level':'template','fid':0. if baseline=='sd15' else 2.}
                    for template in [1,2,3]]})
            per_image,per_seed,summary=b2.summarize_b2_scores(path)
            self.assertEqual(len(per_image),len(rows)); self.assertEqual(len(per_seed),6)
            self.assertTrue(np.allclose(per_seed['bert_f1'],.55))
            self.assertTrue((per_seed[per_seed.baseline=='sd15'].fid==0).all())
            self.assertTrue((b2._output(plan,'sd15')/'per_image_scores.json').is_file())

    @patch('clover.evaluate.inference.load_lora_pipeline')
    @patch('clover.evaluate.inference.load_reference_pipeline')
    def test_vanilla_loader_does_not_install_lora(self, reference, lora):
        import torch
        from clover.evaluate.inference import load_baseline_model
        reference.return_value=MagicMock()
        model=load_baseline_model('sd15',device='cpu')
        reference.assert_called_once(); lora.assert_not_called()
        self.assertIsNone(model.checkpoint); self.assertIsNone(model.variance_module)
        self.assertEqual(model.dtype,torch.float32)


if __name__=='__main__': unittest.main()
