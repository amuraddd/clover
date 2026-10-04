"""CPU-only regression checks for sequential benchmark orchestration."""
import contextlib
import io
import unittest
import weakref
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
                           rollout_chunk_size=8, install_runtime=False),
        ])

    def test_compbench_cli_overrides(self):
        with patch.object(evaluate, 'run_b2_evaluation'), \
             patch.object(evaluate, 'run_compbench_evaluation') as run, \
             patch.object(evaluate, 'load_dotenv'), contextlib.redirect_stdout(io.StringIO()):
            evaluate.main(['--compbench-run-name', 'custom',
                           '--compbench-images-per-prompt', '2', '--compbench-install-runtime'])
        self.assertEqual(run.call_args.kwargs['compbench_run_name'], 'custom')
        self.assertEqual(run.call_args.kwargs['images_per_prompt'], 2)
        self.assertTrue(run.call_args.kwargs['install_runtime'])

    def test_generation_pairing_and_release_before_scoring(self):
        events, references = [], []
        class Model:
            pass
        def load(baseline, **kwargs):
            self.assertEqual(kwargs, {'seed': 123, 'device': 'cuda:0'})
            model = Model()
            model.baseline = baseline
            references.append(weakref.ref(model))
            return model
        def generate(model, **kwargs):
            events.append((model.baseline, kwargs))
            return f"{model.baseline}/{kwargs['run_name']}/manifest.json"
        def score(manifest):
            self.assertTrue(all(ref() is None for ref in references))
            return {'summary': {'color': {'mean': .5}}}
        with patch.object(evaluate, 'setup_compbench') as setup, \
             patch.object(evaluate, 'load_baseline_model', side_effect=load), \
             patch.object(evaluate, 'generate_compbench_images', new=generate), \
             patch.object(evaluate, 'run_compbench_metrics', side_effect=score) as metrics, \
             patch.object(evaluate.torch.cuda, 'empty_cache') as release, \
             contextlib.redirect_stdout(io.StringIO()):
            evaluate.run_compbench_evaluation(['ddpo', 'sd15'], [123, 126], 'run',
                                             rollout_chunk_size=8)
        setup.assert_called_once_with(install_runtime=False)
        self.assertEqual([(b, k['base_seed']) for b, k in events],
                         [('ddpo', 123), ('ddpo', 126), ('sd15', 123), ('sd15', 126)])
        for baseline, kwargs in events:
            self.assertEqual(kwargs['run_name'], f"run_seed_{kwargs['base_seed']}")
            self.assertEqual(kwargs['images_per_prompt'], 10)
            self.assertEqual(kwargs['inference_kwargs'],
                             {'num_inference_steps': 50, 'guidance_scale': 7.5, 'rollout_chunk_size': 8})
        self.assertEqual(metrics.call_count, 4)
        self.assertEqual(release.call_count, 2)

    def test_generation_failure_releases_memory_and_skips_scoring(self):
        with patch.object(evaluate, 'setup_compbench'), \
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
