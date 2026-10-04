"""CPU-only checks for DreamSim statistics, grouping, persistence and orchestration."""
import contextlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from clover.evaluate.b2_evals import b2_evaluation as b2
from clover.evaluate.b2_evals.b2_dreamsim_worker import (
    diversity_contributions, group_records, score_records,
)
from clover.evaluate.b2_run_evaluation import run_b2_evaluation


def records():
    return [dict(baseline='sd15', id=str(i), seed=123, template=1,
                 prompt_index=0, prompt='a cat', image_path=f'{i}.png') for i in range(3)]


class DreamSimTests(unittest.TestCase):
    def test_variance_matches_distinct_pair_cosine_distance(self):
        features = torch.tensor([[2., 0., 0.], [0., 3., 0.], [1., 1., 1.]], dtype=torch.float64)
        normalized = torch.nn.functional.normalize(features, dim=1)
        pair_distances = [1 - torch.dot(normalized[i], normalized[j]).item()
                          for i in range(3) for j in range(i + 1, 3)]
        contributions = diversity_contributions(features)
        self.assertAlmostEqual(np.mean(contributions), np.mean(pair_distances))
        self.assertAlmostEqual(np.mean(contributions), normalized.var(dim=0, correction=1).sum().item())
        self.assertEqual(diversity_contributions(torch.ones(3, 2)), [0., 0., 0.])
        self.assertEqual(diversity_contributions(torch.tensor([[1., 0.], [-1., 0.]])), [2., 2.])

    def test_invalid_embeddings_and_groups(self):
        for features in [torch.ones(1, 2), torch.zeros(2, 2),
                         torch.tensor([[float('nan'), 1.], [1., 0.]])]:
            with self.subTest(features=features), self.assertRaises(ValueError):
                diversity_contributions(features)
        for rows in [[], records()[:1], records() + records(),
                     [records()[0], {**records()[1], 'prompt': 'a dog'}]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                group_records(rows)
        # All group dimensions matter; no cross-prompt/seed/baseline comparisons.
        for key, value in [('baseline', 'ddpo'), ('seed', 124), ('template', 2), ('prompt_index', 1)]:
            rows = records() + [{**r, key: value, 'id': 'other-' + r['id']} for r in records()]
            self.assertEqual(len(group_records(rows)), 2)

    def test_worker_embeds_once_and_preserves_order_across_batches(self):
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.batches = []
            def embed(self, batch):
                self.batches.append(len(batch))
                return batch[:, :, 0, 0]
        def preprocess(image):
            return torch.tensor(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float()
        with TemporaryDirectory() as tmp:
            rows = records()
            for row, color in zip(rows, ['red', 'green', 'blue']):
                row['image_path'] = str(Path(tmp) / row['image_path'])
                Image.new('RGB', (2, 2), color).save(row['image_path'])
            network = Network()
            scored = score_records(rows, network, preprocess, batch_size=2)
        self.assertEqual(network.batches, [2, 1])
        self.assertEqual([r['id'] for r in scored], ['0', '1', '2'])
        for row in scored:
            self.assertAlmostEqual(row['dreamsim_diversity'], 1.)
            self.assertEqual(row['dreamsim_image_count'], 3)

    def test_setup_uses_locked_local_runtime(self):
        with TemporaryDirectory() as tmp, patch.object(b2, 'ROOT', Path(tmp)), patch('subprocess.run') as run:
            b2.setup_b2_dreamsim()
        self.assertEqual(run.call_args_list[0].args[0],
                         ['uv', 'sync', '--locked', '--project', str(b2.DREAMSIM_BACKEND)])

    def test_worker_setup_validation_and_gpu_visibility(self):
        for installed in (True, False):
            for invalid in (None, 'id', 'score', 'count'):
                with self.subTest(installed=installed, invalid=invalid), TemporaryDirectory() as tmp:
                    folder = Path(tmp)
                    backend = folder / 'backend'
                    backend.mkdir()
                    (backend / 'uv.lock').write_text('test lock')
                    python = backend / '.venv/bin/python'
                    if installed:
                        python.parent.mkdir(parents=True)
                        python.touch()
                    rows = records()
                    scored = [{**r, 'dreamsim_diversity': .5, 'dreamsim_contribution': .5,
                               'dreamsim_image_count': 3} for r in rows]
                    if invalid == 'id':
                        scored[0]['id'] = 'wrong'
                    elif invalid == 'score':
                        scored[0]['dreamsim_diversity'] = float('nan')
                    elif invalid == 'count':
                        scored[0]['dreamsim_image_count'] = 1
                    def worker(command, **kwargs):
                        self.assertEqual(command[:2], [str(python), str(b2.DREAMSIM_WORKER)])
                        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '4')
                        self.assertEqual(json.loads(Path(command[2]).read_text())['records'], rows)
                        Path(command[3]).write_text(json.dumps(scored))
                    with patch.object(b2, 'DREAMSIM_BACKEND', backend), \
                         patch.object(b2, '_records', return_value=({}, rows)), \
                         patch.object(b2, '_cached_metric', return_value=None), \
                         patch.object(b2, '_save_metric') as save, \
                         patch.object(b2, '_release'), \
                         patch.object(b2, 'setup_b2_dreamsim') as setup, \
                         patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '4,5'}), \
                         patch('subprocess.run', side_effect=worker):
                        if invalid:
                            with self.assertRaises(ValueError):
                                b2.score_b2_dreamsim(folder / 'plan.json')
                            save.assert_not_called()
                        else:
                            self.assertEqual(b2.score_b2_dreamsim(folder / 'plan.json'), scored)
                            self.assertEqual(save.call_args.args[3], 'dreamsim')
                        self.assertEqual(setup.call_count, 0 if installed else 1)

    def test_cache_skips_worker_and_stale_metadata_is_rejected(self):
        rows = records()
        with patch.object(b2, '_records', return_value=({}, rows)), \
             patch.object(b2, '_cached_metric', return_value=rows), \
             patch('subprocess.run') as run:
            self.assertEqual(b2.score_b2_dreamsim('plan.json'), rows)
            run.assert_not_called()
        with TemporaryDirectory() as tmp:
            plan = {'baselines': ['sd15'], 'records': rows, 'output_root': tmp, 'run_name': 'test'}
            path = Path(tmp) / 'plan.json'
            path.write_text('{}')
            with patch.object(b2, '_image_digest', return_value='images'):
                b2._save_metric(path, plan, rows, 'dreamsim', {'correction': 1})
                self.assertEqual(b2._cached_metric(path, plan, 'dreamsim', {'correction': 1}), rows)
                with self.assertRaisesRegex(ValueError, 'Stale metric cache'):
                    b2._cached_metric(path, plan, 'dreamsim', {'correction': 0})

    def test_pipeline_saves_dreamsim_before_summary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / 'plan.json'
            plan.write_text('{}')
            folder = root / 'clover/evaluate/metrics/test'
            def score(*args, **kwargs):
                self.assertTrue((folder / 'b2_imagereward.json').is_file())
                return [{'dreamsim_diversity': .4}]
            def summarize(*args, **kwargs):
                self.assertEqual(json.loads((folder / 'b2_dreamsim.json').read_text()),
                                 [{'dreamsim_diversity': .4}])
                frame = pd.DataFrame([{'dreamsim_diversity': .4}])
                return frame, frame, frame
            with patch.object(b2, 'ROOT', root), patch.multiple(
                b2, prepare_b2_run=Mock(return_value=plan), generate_b2_images=Mock(return_value={}),
                score_b2_fid=Mock(return_value=[]), score_b2_clip=Mock(return_value=[]),
                caption_b2_images=Mock(return_value=[]), score_b2_bert=Mock(return_value=[]),
                score_b2_imagereward=Mock(return_value=[]), score_b2_dreamsim=score,
                summarize_b2_scores=summarize), patch('torch.cuda.empty_cache'), \
                 contextlib.redirect_stdout(io.StringIO()):
                run_b2_evaluation(['sd15'], [123], 'test', 3, .8)
                with patch.object(b2, 'score_b2_dreamsim', side_effect=RuntimeError('worker failed')):
                    with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                        run_b2_evaluation(['sd15'], [123], 'partial', 3, .8)
                self.assertTrue((root / 'clover/evaluate/metrics/partial/b2_imagereward.json').is_file())
                self.assertFalse((root / 'clover/evaluate/metrics/partial/b2_dreamsim.json').exists())
        with patch.object(b2, 'prepare_b2_run') as prepare:
            with self.assertRaisesRegex(ValueError, 'images_per_prompt'):
                run_b2_evaluation(['sd15'], [123], 'invalid', 1, .8)
            prepare.assert_not_called()

    def test_summary_weights_prompts_equally_and_reports_seed_std(self):
        with TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / 'plan.json'
            plan_path.write_text('{}')
            plan = dict(baselines=['sd15'], seeds=[123, 124], output_root=tmp,
                        run_name='test', inference={})
            rows = []
            for seed in plan['seeds']:
                for template in [1, 2, 3]:
                    for prompt_index, count in [(0, 2), (1, 4)]:
                        # Means by prompt: (0.2, 0.6) then (0.4, 0.8).
                        value = .2 + .4 * prompt_index + .2 * (seed - 123)
                        for replicate in range(count):
                            rows.append(dict(baseline='sd15', seed=seed, template=template,
                                template_name=b2.TEMPLATES[template], prompt_index=prompt_index,
                                id=f'{seed}:{template}:{prompt_index}:{replicate}',
                                clip=.3, caption='cat', bert_precision=.5, bert_recall=.5,
                                bert_f1=.5, imagereward=1., dreamsim_diversity=value,
                                dreamsim_contribution=value, dreamsim_image_count=count))
            plan['records'] = rows
            folder = b2._output(plan, 'sd15')
            with patch.object(b2, '_records', return_value=(plan, rows)), \
                 patch.object(b2, '_image_digest', return_value='images'):
                for metric in ['clip', 'bert', 'imagereward', 'dreamsim']:
                    b2._save_metric(plan_path, plan, list(reversed(rows)), metric, {})
                b2._save(folder / 'fid.json', dict(plan_sha256=b2._hash(plan_path),
                    images_sha256='images', reference_images_sha256='images', records=[
                        dict(baseline='sd15', seed=seed, template=template, fid=0., level='template')
                        for seed in plan['seeds'] for template in [1, 2, 3]]))
                _, per_seed, summary = b2.summarize_b2_scores(plan_path)
                np.testing.assert_allclose(per_seed['dreamsim_diversity'], [.4] * 3 + [.6] * 3)
                np.testing.assert_allclose(summary['dreamsim_diversity_mean'], .5)
                np.testing.assert_allclose(summary['dreamsim_diversity_std'], .2 / np.sqrt(2))
                with patch.object(b2, '_records', return_value=({**plan, 'baselines': ['sd15', 'ddpo']}, rows)):
                    with self.assertRaisesRegex(ValueError, 'Incomplete DreamSim'):
                        b2.summarize_b2_scores(plan_path)


if __name__ == '__main__':
    unittest.main()
