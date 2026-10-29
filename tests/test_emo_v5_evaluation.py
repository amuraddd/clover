"""CPU-only periodic evaluation checks; no model downloads or generation."""
import ast
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

from clover.evaluate import emo_v5_metrics as metrics


class EMOV5EvaluationTests(unittest.TestCase):
    def test_real_fid_dependency_compatibility(self):
        # Exercise the installed SciPy API, without models or downloads.
        from pytorch_fid.fid_score import calculate_frechet_distance

        mean = np.zeros(2)
        covariance = np.diag([1., 4.])
        self.assertAlmostEqual(
            calculate_frechet_distance(mean, covariance, mean, covariance), 0.)
        # Mean displacement contributes 5; covariance displacement contributes 2.
        self.assertAlmostEqual(calculate_frechet_distance(
            mean, covariance, np.array([1., 2.]), np.diag([4., 9.])), 7.)
        singular = np.diag([1., 0.])
        self.assertAlmostEqual(
            calculate_frechet_distance(mean, singular, mean, singular), 0.)

    def test_reference_cache_and_settings_invalidation(self):
        config = SimpleNamespace(height=16, width=16, num_inference_steps=2,
                                 guidance_scale=5., negative_prompt='bad')
        reference = Mock()
        reference.scheduler.config = {}
        reference.return_value.images = [Image.new('RGB', (16, 16)) for _ in range(10)]
        with TemporaryDirectory() as temp, patch.object(metrics, 'ROOT', Path(temp)), \
                patch('diffusers.StableDiffusionPipeline.from_pretrained', return_value=reference) as load:
            paths, metadata = metrics.sd15_reference_images(config, ['cat'], torch.device('cpu'), torch.float32)
            self.assertEqual(len(paths), 10)
            self.assertTrue(all(p.exists() for p in paths))
            metrics.sd15_reference_images(config, ['cat'], torch.device('cpu'), torch.float32)
            self.assertEqual(load.call_count, 1)
            config.guidance_scale = 6.
            changed, _ = metrics.sd15_reference_images(config, ['cat'], torch.device('cpu'), torch.float32)
            self.assertNotEqual(paths, changed)
            self.assertEqual(load.call_count, 2)
            changed[0].unlink()
            metrics.sd15_reference_images(config, ['cat'], torch.device('cpu'), torch.float32)
            self.assertEqual(load.call_count, 3)
            self.assertEqual(metadata['model_id'], metrics.REFERENCE_MODEL)
            seeded, seeded_metadata = metrics.sd15_reference_images(
                config, ['cat'], torch.device('cpu'), torch.float32, eval_seed=456)
            self.assertNotEqual(seeded, changed)
            self.assertEqual(seeded_metadata['seed'], 456)
            self.assertEqual(reference.call_args.kwargs['generator'].initial_seed(), 456)
            metrics.sd15_reference_images(config, ['cat'], torch.device('cpu'), torch.float32, eval_seed=456)
            self.assertEqual(load.call_count, 4)

    def test_fid_uses_only_supplied_images_and_pools_all_prompts(self):
        paths = [Path(f'image_{i}.png') for i in range(20)]
        refs = [Path(f'ref_{i}.png') for i in range(20)]
        with patch.object(metrics, 'sd15_reference_images', return_value=(refs, {})) as reference, \
                patch('pytorch_fid.inception.InceptionV3') as inception, \
                patch('pytorch_fid.fid_score.calculate_activation_statistics', return_value=(np.zeros(2), np.eye(2))) as stats, \
                patch('pytorch_fid.fid_score.calculate_frechet_distance', return_value=3.) as fid, \
                patch.object(metrics, 'dreamsim_diversity', return_value={'mean': .2}) as diversity:
            inception.BLOCK_INDEX_BY_DIM = {2048: 3}
            result = metrics.evaluate_fid_and_diversity(None, ['cat', 'dog'], paths, torch.device('cpu'), torch.float32, Path('.'), eval_seed=456)
            reference.assert_called_once_with(None, ['cat', 'dog'], torch.device('cpu'), torch.float32, eval_seed=456)
            self.assertEqual(result['fid'], 3.)
            self.assertEqual(result['dreamsim_diversity'], .2)
            self.assertEqual(stats.call_args_list[0].args[0], list(map(str, paths)))
            self.assertEqual(stats.call_args_list[1].args[0], list(map(str, refs)))
            records = diversity.call_args.args[0]
            self.assertEqual({r['seed'] for r in records}, {456})
            self.assertEqual([r['prompt'] for r in records], ['cat'] * 10 + ['dog'] * 10)
            fid.return_value = float('nan')
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                metrics.evaluate_fid_and_diversity(None, ['cat', 'dog'], paths, torch.device('cpu'), torch.float32, Path('.'))
            with self.assertRaisesRegex(ValueError, 'exactly ten'):
                metrics.evaluate_fid_and_diversity(None, ['cat'], paths, torch.device('cpu'), torch.float32, Path('.'))

    def test_dreamsim_grouping_and_worker_device(self):
        records = [dict(id=str(i), prompt_index=i // 10, prompt=['cat', 'dog'][i // 10]) for i in range(20)]
        def worker(command, **kwargs):
            request = json.loads(Path(command[2]).read_text())
            scored = [dict(r, dreamsim_diversity=.2 + .4 * r['prompt_index'], dreamsim_image_count=10)
                      for r in request['records']]
            Path(command[3]).write_text(json.dumps(scored))
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-b')
            self.assertEqual(command[-1], 'cuda:0')
        with TemporaryDirectory() as temp, patch.object(metrics, 'setup_b2_dreamsim'), \
                patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': 'GPU-a,GPU-b'}), \
                patch.object(metrics.subprocess, 'run', side_effect=worker):
            result = metrics.dreamsim_diversity(records, Path(temp), torch.device('cuda:1'))
            self.assertAlmostEqual(result['mean'], .4)
            self.assertEqual([r['count'] for r in result['per_prompt']], [10, 10])

    def test_evaluator_generates_ten_per_prompt_and_preserves_rewards(self):
        source = Path('clover/baselines/emo_v5.py')
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_evaluate_emo_v5')
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
        generate = Mock(side_effect=lambda pipe, prompts, *args, **kw: [Image.new('RGB', (8, 8)) for _ in prompts])
        def save(path, payload):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
        namespace = dict(Path=Path, np=np, math=math, json=json,
                         standard_eval_prompts=lambda c: ['cat', 'dog'],
                         _generate_emo_v5_eval_images=generate, save_json=save,
                         save_image_grid_outputs=lambda images, prompts, folder, *a, **kw:
                         [dict(image=str(folder / f'{i}.png')) for i in range(len(images))])
        exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
        with TemporaryDirectory() as temp, patch.object(metrics, 'evaluation_data_dir', return_value=Path(temp) / 'data'), \
                patch.object(metrics, 'evaluate_fid_and_diversity', return_value=dict(fid=1., dreamsim_diversity=.3)) as score, \
                patch('clover.utils.rewards_utils.clip_reward', return_value=torch.ones(20)), \
                patch('clover.utils.rewards_utils.bert_reward', return_value=torch.ones(20)):
            config = SimpleNamespace(output_dir=temp)
            namespace['_evaluate_emo_v5'](None, config, torch.device('cpu'), torch.float32, epoch=2)
            self.assertEqual([c.args[1] for c in generate.call_args_list], [['cat'] * 10, ['dog'] * 10] * 3)
            self.assertEqual([c.kwargs['seed'] for c in generate.call_args_list], [123, 124, 456, 457, 789, 790])
            results = json.loads((Path(temp) / 'evals/eval_metrics.json').read_text())
            self.assertEqual([r['eval_seed'] for r in results], [123, 456, 789])
            self.assertEqual([r['prompt_seeds'] for r in results], [[123, 124], [456, 457], [789, 790]])
            self.assertEqual([c.kwargs['eval_seed'] for c in score.call_args_list], [123, 456, 789])
            image_sets = [set(r['image_paths']) for r in results]
            self.assertEqual(len(set.union(*image_sets)), 60)
            for seed, row in zip(metrics.EVAL_SEEDS, results):
                self.assertTrue(all(f'/seed_{seed}/images/' in p for p in row['image_paths']))
                self.assertEqual(row['clip_reward_summary']['count'], 20)
                self.assertIn('fid', row)
                self.assertIn('dreamsim_diversity', row)
            result = results[0]
            self.assertEqual(result['samples_per_prompt'], 10)
            self.assertEqual(result['clip_reward_summary']['count'], 20)
            self.assertEqual(len(result['bert_reward']), 20)
            self.assertEqual(score.call_args.args[1], ['cat', 'dog'])
            self.assertEqual(result['fid'], 1.)


if __name__ == '__main__':
    unittest.main()
