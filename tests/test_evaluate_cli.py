"""CPU-only regression checks for sequential benchmark orchestration."""
import contextlib
import io
import unittest
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call, patch

import evaluate


class EvaluationCliTests(unittest.TestCase):
    def test_b2_then_compbench_with_shared_baselines_and_seeds(self):
        stages = MagicMock()
        with patch.object(evaluate, 'run_b2_evaluation', stages.b2), \
             patch.object(evaluate, 'run_compbench_evaluation', stages.compbench), \
             patch.object(evaluate, 'load_dotenv'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(evaluate.main([
                '--baselines', 'emo_v5', 'sd15', '--seeds', '124', '126',
                '--b2-run-name', 'paired', '--rollout-chunk-size', '8',
            ]), 0)
        self.assertEqual(stages.mock_calls, [
            call.b2(baselines=['emo_v5', 'sd15'], seeds=[124, 126],
                    b2_run_name='paired', images_per_prompt=20, fraction=.8, rollout_chunk_size=8),
            call.compbench(baselines=['emo_v5', 'sd15'], seeds=[124, 126],
                           compbench_run_name='paired', images_per_prompt=10,
                           rollout_chunk_size=8, install_runtime=False, prompts_per_category=None),
        ])

    def test_compbench_cli_overrides(self):
        with patch.object(evaluate, 'run_b2_evaluation'), \
             patch.object(evaluate, 'run_compbench_evaluation') as run, \
             patch.object(evaluate, 'load_dotenv'), contextlib.redirect_stdout(io.StringIO()):
            evaluate.main(['--compbench-run-name', 'custom',
                           '--compbench-images-per-prompt', '2', '--compbench-install-runtime',
                           '--compbench-prompts-per-category', '5'])
        self.assertEqual(run.call_args.kwargs['compbench_run_name'], 'custom')
        self.assertEqual(run.call_args.kwargs['images_per_prompt'], 2)
        self.assertTrue(run.call_args.kwargs['install_runtime'])
        self.assertEqual(run.call_args.kwargs['prompts_per_category'], 5)

    def test_prompt_limit_defaults_and_validation(self):
        parser = evaluate.build_parser()
        self.assertIsNone(parser.parse_args([]).compbench_prompts_per_category)
        for value in ('0', '-1', '1.5'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(['--compbench-prompts-per-category', value])

    def test_prompt_limit_shared_by_cache_lookup_and_generation(self):
        prompts = {'color': ['red', 'blue', 'green'], 'shape': ['round']}
        for limit in (None, 2, 10):
            with self.subTest(limit=limit), \
                 patch.object(evaluate, 'setup_compbench'), \
                 patch.object(evaluate, 'load_compbench_prompts', return_value=prompts) as load_prompts, \
                 patch.object(evaluate, 'find_compbench_images', side_effect=['sd15/cached', None]) as find, \
                 patch.object(evaluate, 'load_baseline_model'), \
                 patch.object(evaluate, 'generate_compbench_images', return_value='ddpo/generated') as generate, \
                 patch.object(evaluate, 'run_compbench_metrics', return_value={'summary': {}}), \
                 patch.object(evaluate, 'run_compbench_distribution_metrics', return_value={}), \
                 patch.object(evaluate.torch.cuda, 'empty_cache'), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate.run_compbench_evaluation(['ddpo', 'sd15'], [123], 'limited',
                                                 prompts_per_category=limit)
            expected = None if limit is None else {k: v[:limit] for k, v in prompts.items()}
            for lookup in find.call_args_list:
                self.assertEqual(lookup.kwargs['prompts_by_category'], expected)
            self.assertEqual(generate.call_args.kwargs['prompts_by_category'], expected)
            if limit is None:
                load_prompts.assert_not_called()
            else:
                load_prompts.assert_called_once_with(categories=tuple(evaluate.COMPBENCH_METRICS))

    def test_setup_installs_missing_runtime_or_weights_before_loading_images(self):
        for runtime_exists, weights_exist, force in (
            (False, False, False), (False, True, False),
            (True, False, False), (True, True, False), (True, True, True),
        ):
            with self.subTest(runtime=runtime_exists, weights=weights_exist, force=force), \
                 TemporaryDirectory() as directory:
                root = Path(directory)
                runtime = root / 'backend/.venv/bin/python'
                weights = root / 'data/upstream/UniDet_eval/experts/expert_weights/Unified_learned_OCIM_RS200_6x+2x.pth'
                for path, exists in ((runtime, runtime_exists), (weights, weights_exist)):
                    if exists:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.touch()
                stages = MagicMock()
                stages.find.return_value = 'cached/manifest.json'
                stages.score.return_value = {'summary': {}}
                stages.distribution.return_value = {}
                with patch.object(evaluate, 'COMPBENCH_BACKEND', root / 'backend'), \
                     patch.object(evaluate, 'COMPBENCH_DATA_ROOT', root / 'data'), \
                     patch.object(evaluate, 'setup_compbench', stages.setup), \
                     patch.object(evaluate, 'find_compbench_images', stages.find), \
                     patch.object(evaluate, 'load_baseline_model') as load, \
                     patch.object(evaluate, 'run_compbench_metrics', stages.score), \
                     patch.object(evaluate, 'run_compbench_distribution_metrics', stages.distribution), \
                     contextlib.redirect_stdout(io.StringIO()):
                    evaluate.run_compbench_evaluation(['sd15'], [123], 'run', install_runtime=force)
                self.assertEqual(stages.mock_calls[0], call.setup(
                    install_runtime=force or not runtime_exists or not weights_exist))
                self.assertEqual(stages.mock_calls[1][0], 'find')
                load.assert_not_called()

    def test_setup_failure_stops_before_generation_or_scoring(self):
        with patch.object(evaluate, 'setup_compbench', side_effect=RuntimeError('setup failed')), \
             patch.object(evaluate, 'find_compbench_images') as find, \
             patch.object(evaluate, 'load_baseline_model') as load, \
             patch.object(evaluate, 'run_compbench_metrics') as score, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'setup failed'):
                evaluate.run_compbench_evaluation(['sd15'], [123], 'run')
        find.assert_not_called()
        load.assert_not_called()
        score.assert_not_called()

    def test_generation_pairing_and_release_before_scoring(self):
        events, references, stages = [], [], []
        class Model:
            pass
        def load(baseline, **kwargs):
            stages.append(('load', baseline))
            self.assertEqual(kwargs, {'seed': 123, 'device': 'cuda:0'})
            model = Model()
            model.baseline = baseline
            references.append(weakref.ref(model))
            return model
        def generate(model, **kwargs):
            events.append((model.baseline, kwargs))
            return f"{model.baseline}/{kwargs['run_name']}/manifest.json"
        def distribution_score(manifests):
            self.assertTrue(all(ref() is None for ref in references))
            stages.append(('distribution', manifests[-1]))
            return {}
        def score(manifest, *, resume):
            self.assertTrue(resume)
            stages.append(('official', manifest))
            self.assertTrue(all(ref() is None for ref in references))
            return {'summary': {'color': {'mean': .5}}}
        with patch.object(evaluate, 'setup_compbench') as setup, \
             patch.object(evaluate, 'find_compbench_images', return_value=None), \
             patch.object(evaluate, 'load_baseline_model', side_effect=load), \
             patch.object(evaluate, 'generate_compbench_images', new=generate), \
             patch.object(evaluate, 'run_compbench_metrics', side_effect=score) as metrics, \
             patch.object(evaluate, 'run_compbench_distribution_metrics', side_effect=distribution_score) as distribution, \
             patch.object(evaluate.torch.cuda, 'empty_cache') as release, \
             contextlib.redirect_stdout(io.StringIO()):
            evaluate.run_compbench_evaluation(['ddpo', 'sd15'], [123, 126], 'run',
                                             rollout_chunk_size=8, install_runtime=True)
        setup.assert_called_once_with(install_runtime=True)
        self.assertEqual([(b, k['base_seed']) for b, k in events],
                         [('sd15', 123), ('sd15', 126), ('ddpo', 123), ('ddpo', 126)])
        for baseline, kwargs in events:
            self.assertEqual(kwargs['run_name'], f"run_seed_{kwargs['base_seed']}")
            self.assertEqual(kwargs['images_per_prompt'], 10)
            self.assertEqual(kwargs['inference_kwargs'],
                             {'num_inference_steps': 50, 'guidance_scale': 5.0, 'rollout_chunk_size': 8})
        self.assertEqual(distribution.call_args_list, [
            call(['sd15/run_seed_123/manifest.json']),
            call(['sd15/run_seed_126/manifest.json']),
            call(['sd15/run_seed_123/manifest.json', 'ddpo/run_seed_123/manifest.json']),
            call(['sd15/run_seed_126/manifest.json', 'ddpo/run_seed_126/manifest.json']),
        ])
        self.assertEqual(stages, [
            ('load', 'sd15'),
            ('official', 'sd15/run_seed_123/manifest.json'),
            ('distribution', 'sd15/run_seed_123/manifest.json'),
            ('official', 'sd15/run_seed_126/manifest.json'),
            ('distribution', 'sd15/run_seed_126/manifest.json'),
            ('load', 'ddpo'),
            ('official', 'ddpo/run_seed_123/manifest.json'),
            ('distribution', 'ddpo/run_seed_123/manifest.json'),
            ('official', 'ddpo/run_seed_126/manifest.json'),
            ('distribution', 'ddpo/run_seed_126/manifest.json'),
        ])
        self.assertEqual(metrics.call_count, 4)
        self.assertEqual(release.call_count, 2)

    def test_resume_skips_complete_seeds_and_loads_only_missing_baseline(self):
        def find(baseline, *, base_seed, **kwargs):
            return None if (baseline, base_seed) == ('ddpo', 126) else f'{baseline}/{base_seed}'
        with patch.object(evaluate, 'setup_compbench'), \
             patch.object(evaluate, 'find_compbench_images', side_effect=find), \
             patch.object(evaluate, 'load_baseline_model') as load, \
             patch.object(evaluate, 'generate_compbench_images', return_value='ddpo/126') as generate, \
             patch.object(evaluate, 'run_compbench_metrics', return_value={'summary': {}}) as official, \
             patch.object(evaluate, 'run_compbench_distribution_metrics', return_value={}) as distribution, \
             patch.object(evaluate.torch.cuda, 'empty_cache'), \
             contextlib.redirect_stdout(io.StringIO()):
            evaluate.run_compbench_evaluation(['ddpo', 'sd15'], [123, 126], 'run')
        load.assert_called_once_with('ddpo', seed=123, device='cuda:0')
        generate.assert_called_once()
        self.assertEqual(generate.call_args.kwargs['base_seed'], 126)
        self.assertTrue(generate.call_args.kwargs['resume'])
        self.assertEqual(official.call_args_list, [call(p, resume=True) for p in
                         ('sd15/123', 'sd15/126', 'ddpo/123', 'ddpo/126')])
        self.assertEqual(distribution.call_count, 4)

    def test_generation_failure_releases_memory_and_skips_scoring(self):
        with patch.object(evaluate, 'setup_compbench'), \
             patch.object(evaluate, 'find_compbench_images', return_value=None), \
             patch.object(evaluate, 'load_baseline_model'), \
             patch.object(evaluate, 'generate_compbench_images', side_effect=RuntimeError('failed')), \
             patch.object(evaluate, 'run_compbench_metrics') as score, \
             patch.object(evaluate.torch.cuda, 'empty_cache') as release, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                evaluate.run_compbench_evaluation(['sd15'], [123], 'run')
        release.assert_called_once()
        score.assert_not_called()


if __name__ == '__main__':
    unittest.main()
