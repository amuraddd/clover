import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from clover.evaluate import t21_comp_bench as compbench

generate_compbench_images = compbench.generate_compbench_images
collect_compbench_metrics = compbench.collect_compbench_metrics
run_compbench_metrics = compbench.run_compbench_metrics
load_compbench_prompts = compbench.load_compbench_prompts
REVISION = compbench.REVISION



class CompBenchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls = []
        def generate(prompts, **kwargs):
            self.calls.append((prompts, kwargs))
            return [Image.new('RGB', (8, 8))]
        self.model = SimpleNamespace(baseline='emo_v4', checkpoint=self.root / 'checkpoint.pt',
            config=SimpleNamespace(height=8, width=8, num_inference_steps=2,
                                   guidance_scale=7.5, negative_prompt=''), generate=generate)

    def generate(self, **kwargs):
        return generate_compbench_images(self.model, data_root=self.root,
            prompts_by_category={'color': ['a red dog and a blue cat', 'a green bench and a red bowl']},
            images_per_prompt=2, **kwargs)

    def write_scores(self, path, rows):
        out = path.parent / 'color/annotation_blip/vqa_result.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows))

    def test_seed_pairing_filenames_and_id_join(self):
        path = self.generate(base_seed=50)
        self.assertEqual([c[1]['seed'] for c in self.calls], [50, 51, 50, 51])
        records = json.loads(path.read_text())['records']
        self.assertEqual([r['question_id'] for r in records], list(range(4)))
        for record in records:
            self.assertTrue((path.parent / record['image']).is_file())
            self.assertEqual(Path(record['image']).stem.split('_')[0], record['prompt'])
        self.write_scores(path, [{'question_id': i, 'answer': str(i / 4)} for i in reversed(range(4))])
        result = collect_compbench_metrics(path, output_root=self.root / 'outputs')
        self.assertEqual(result['summary']['color']['mean'], 0.375)
        self.assertEqual([r['score'] for r in result['per_image']], [0, 0.25, 0.5, 0.75])
        self.assertTrue((self.root / 'outputs/emo_v4/evals/t2i_compbench/default/metrics.json').is_file())

    def test_sd15_without_checkpoint(self):
        self.model.baseline = 'sd15'
        self.model.checkpoint = None
        path = self.generate()
        self.assertIsNone(json.loads(path.read_text())['checkpoint'])
        self.write_scores(path, [{'question_id': i, 'answer': .5} for i in range(4)])
        result = collect_compbench_metrics(path, output_root=self.root / 'outputs')
        self.assertIsNone(result['checkpoint'])
        self.assertTrue((self.root / 'outputs/sd15/evals/t2i_compbench/default/metrics.json').is_file())

    def test_refuse_mixed_runs_and_unsafe_prompts(self):
        self.generate()
        with self.assertRaises(FileExistsError):
            self.generate()
        with self.assertRaisesRegex(ValueError, 'filename'):
            generate_compbench_images(self.model, data_root=self.root, prompts_by_category={'color': ['bad_prompt']})

    def test_reject_missing_duplicate_and_nonfinite_scores(self):
        path = self.generate()
        rows = [{'question_id': i, 'answer': .5} for i in range(4)]
        for bad in (rows[:-1], rows + [rows[0]], [dict(r, answer='nan') for r in rows]):
            self.write_scores(path, bad)
            with self.assertRaises(ValueError):
                collect_compbench_metrics(path, output_root=self.root)

    def test_partial_generation_cannot_be_scored(self):
        def fail(*args, **kwargs):
            raise RuntimeError('generation failed')
        self.model.generate = fail
        with self.assertRaises(RuntimeError):
            self.generate()
        path = self.root / 'emo_v4/t2i_compbench/default/manifest.json'
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            collect_compbench_metrics(path)

    def test_official_runner_arguments_and_gpu_limit(self):
        path = self.generate()
        upstream = self.root / 'upstream'
        upstream.mkdir()
        (upstream / 'clover_source.json').write_text(json.dumps({'revision': REVISION}))
        def run(command, **kwargs):
            self.assertEqual(command[-3:], ['BLIP_vqa.py', '--out_dir', str(path.parent / 'color')])
            self.assertEqual(kwargs['cwd'], upstream / 'BLIPvqa_eval')
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '2')
            self.write_scores(path, [{'question_id': i, 'answer': .5} for i in range(4)])
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '2,3'}, clear=True), \
             patch('clover.evaluate.t21_comp_bench.subprocess.run', side_effect=run):
            result = run_compbench_metrics(path, upstream_dir=upstream,
                python_command=['python'], output_root=self.root)
        self.assertEqual(result['summary']['color']['mean'], .5)

    def test_find_completed_images_checks_settings_and_files(self):
        path = self.generate()
        prompts = {'color': ['a red dog and a blue cat', 'a green bench and a red bowl']}
        def find(**kwargs):
            return compbench.find_compbench_images('emo_v4', run_name='default',
                base_seed=123, images_per_prompt=2, data_root=self.root,
                prompts_by_category=prompts, **kwargs)
        self.assertEqual(find(inference_kwargs={'guidance_scale': 7.5}), path)
        with self.assertRaisesRegex(ValueError, 'guidance_scale'):
            find(inference_kwargs={'guidance_scale': 5.})
        manifest = json.loads(path.read_text())
        (path.parent / manifest['records'][0]['image']).unlink()
        with self.assertRaises(FileNotFoundError):
            find()

    def test_interrupted_generation_preserves_images_and_resumes_missing(self):
        original = self.model.generate
        attempts = []
        def interrupt(*args, **kwargs):
            attempts.append(kwargs)
            if len(attempts) == 4:
                raise RuntimeError('interrupted')
            return original(*args, **kwargs)
        self.model.generate = interrupt
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.generate()
        run = self.root / 'emo_v4/t2i_compbench/default'
        before = {p: p.stat().st_mtime_ns for p in run.glob('color/samples/*.png')}
        self.assertEqual(len(before), 3)
        self.assertIsNone(compbench.find_compbench_images('emo_v4', run_name='default',
            base_seed=123, images_per_prompt=2, data_root=self.root))
        self.model.generate = original
        self.calls.clear()
        path = self.generate(resume=True)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})
        self.assertTrue(json.loads(path.read_text())['complete'])
        self.assertEqual(len(json.loads(path.read_text())['records']), 4)
        self.calls.clear()
        self.generate(resume=True)
        self.assertEqual(self.calls, [])

    def test_resume_official_scores_without_runtime_and_reject_invalid_cache(self):
        path = self.generate()
        self.write_scores(path, [{'question_id': i, 'answer': .5} for i in range(4)])
        with patch.object(compbench.subprocess, 'run') as run:
            result = run_compbench_metrics(path, resume=True,
                upstream_dir=self.root / 'absent', output_root=self.root / 'outputs')
            run.assert_not_called()
        self.assertEqual(result['summary']['color']['mean'], .5)
        self.write_scores(path, [{'question_id': 0, 'answer': .5}])
        with self.assertRaisesRegex(ValueError, 'Score IDs'):
            run_compbench_metrics(path, resume=True, upstream_dir=self.root / 'absent')

    def test_resume_only_missing_official_category(self):
        path = generate_compbench_images(self.model, data_root=self.root,
            prompts_by_category={'color': ['a red dog'], 'shape': ['a round bowl']},
            images_per_prompt=2)
        rows = [{'question_id': i, 'answer': .5} for i in range(2)]
        self.write_scores(path, rows)
        upstream = self.root / 'upstream'
        upstream.mkdir()
        (upstream / 'clover_source.json').write_text(json.dumps({'revision': REVISION}))
        def score(command, **kwargs):
            self.assertEqual(command[-1], str(path.parent / 'shape'))
            result = path.parent / 'shape/annotation_blip/vqa_result.json'
            result.parent.mkdir(parents=True)
            result.write_text(json.dumps(rows))
        with patch.object(compbench.subprocess, 'run', side_effect=score) as run:
            result = run_compbench_metrics(path, resume=True, upstream_dir=upstream,
                python_command=['python'], output_root=self.root / 'outputs')
        self.assertEqual(run.call_count, 1)
        self.assertEqual(set(result['summary']), {'color', 'shape'})

    def test_unidet_and_unsupported_category_dispatch(self):
        _metric_spec = compbench._metric_spec
        cwd, args, output = _metric_spec('spatial', self.root, self.root / 'spatial')
        self.assertEqual(cwd.name, 'UniDet_eval')
        self.assertEqual(args[0], '2D_spatial_eval.py')
        self.assertTrue(str(output).endswith('labels/annotation_obj_detection_2d/vqa_result.json'))
        with self.assertRaisesRegex(ValueError, 'CLIPScore'):
            _metric_spec('non_spatial', self.root, self.root)

    def test_prompt_loader_is_explicit_about_download(self):
        with self.assertRaises(FileNotFoundError):
            load_compbench_prompts(data_root=self.root)
        with self.assertRaises(ValueError):
            load_compbench_prompts(categories=['made_up'], data_root=self.root)


if __name__ == '__main__':
    unittest.main()
