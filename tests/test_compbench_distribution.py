"""CPU checks for adapting CompBench to the shared B2 metric pipeline."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from clover.evaluate import t21_comp_bench as cb
from clover.evaluate.b2_evals import b2_evaluation as b2
from clover.evaluate.b2_evals.b2_dreamsim_worker import group_records


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = []
        for baseline in ('ddpo', 'sd15'):
            model = SimpleNamespace(baseline=baseline, checkpoint=None,
                config=SimpleNamespace(height=8, width=8, num_inference_steps=2,
                                       guidance_scale=5., negative_prompt=''),
                generate=lambda *args, **kwargs: [Image.new('RGB', (8, 8))])
            self.paths.append(cb.generate_compbench_images(model,
                prompts_by_category={'color': ['a red dog', 'a blue cat'], 'shape': ['a round bowl']},
                data_root=self.root, images_per_prompt=2))

    def prepare(self):
        return cb.prepare_compbench_distribution_plan(self.paths, output_root=self.root / 'outputs')

    def test_grouping_and_hash_validation(self):
        path = self.prepare()
        plan, rows = b2._records(path)
        self.assertEqual(len(group_records(rows)), 6)
        self.assertEqual({r['seed'] for r in rows}, {123})
        self.assertEqual({r['image_seed'] for r in rows}, {123, 124})
        self.assertEqual({len(indices) for indices in group_records(rows).values()}, {2})
        self.assertEqual(path, self.prepare())
        Image.new('RGB', (8, 8), 'red').save(rows[0]['image_path'])
        with self.assertRaisesRegex(ValueError, 'images changed'):
            self.prepare()
        with self.assertRaisesRegex(ValueError, 'Image changed'):
            b2._records(path)

    def test_reject_unpaired_or_incomplete_replicates(self):
        with self.assertRaisesRegex(ValueError, 'sd15'):
            cb.prepare_compbench_distribution_plan(self.paths[:1])
        original = json.loads(self.paths[0].read_text())
        for field, value in (('base_seed', 456), ('records', original['records'][:-1])):
            changed = dict(original, **{field: value})
            self.paths[0].write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                self.prepare()
        self.paths[0].write_text(json.dumps(original))
        changed = json.loads(self.paths[1].read_text())
        changed['records'].reverse()
        self.paths[1].write_text(json.dumps(changed))
        self.prepare()  # pairing follows IDs rather than input order

    def test_shared_fid_uses_category_groups_and_reuses_features(self):
        reference_path = cb.prepare_compbench_distribution_plan(
            [self.paths[1]], output_root=self.root / 'outputs')
        path = self.prepare()
        # Preparing the pair must not invalidate the standalone reference plan.
        b2._records(reference_path)
        weights = self.root / 'inception.pth'
        weights.write_bytes(b'test weights')
        with patch.object(b2, '_load_local_inception') as network, \
             patch.object(b2, '_release'), \
             patch('clover.utils.diversity_score._inception_outputs',
                   side_effect=lambda images, **kwargs: torch.ones(len(images), 2048)) as embed, \
             patch('clover.utils.diversity_score.frechet_inception_distance', return_value=1.25) as distance:
            b2.score_b2_fid(reference_path, device='cpu', inception_weights=weights)
            embed.reset_mock()
            scores = b2.score_b2_fid(path, device='cpu', inception_weights=weights)
            self.assertEqual(embed.call_count, 1)  # only ddpo; sd15 features are reused
            self.assertEqual(len(scores), 4)
            self.assertEqual([c.args[0].shape for c in distance.call_args_list], [(4, 2048), (2, 2048)])
            self.assertEqual([r['fid'] for r in scores if r['baseline'] == 'sd15'], [0., 0.])
            embed.reset_mock()
            b2.score_b2_fid(path, device='cpu', inception_weights=weights)
            embed.assert_not_called()
        saved = json.loads((self.root / 'outputs/ddpo/evals/t2i_compbench/default/fid.json').read_text())
        self.assertEqual(saved['metadata']['reference'], 'sd15')
        self.assertIn('weights_sha256', saved['metadata'])

    def test_summary_uses_shared_dreamsim_and_persists(self):
        def fid(path, **kwargs):
            plan, _ = b2._records(path)
            return [{'baseline': b, 'template': c, 'fid': 0. if b == 'sd15' else 2.}
                    for b in plan['baselines'] for c in plan['templates']]
        def dreamsim(path, **kwargs):
            _, rows = b2._records(path)
            return [dict(r, dreamsim_diversity=.2 + .2 * r['prompt_index']) for r in rows]
        with patch.object(b2, 'score_b2_fid', side_effect=fid), \
             patch.object(b2, 'score_b2_dreamsim', side_effect=dreamsim):
            result = cb.run_compbench_distribution_metrics(self.paths, output_root=self.root / 'outputs', device='cpu')
        self.assertAlmostEqual(result['ddpo']['color']['dreamsim_diversity'], .3)
        self.assertEqual(result['ddpo']['shape']['fid'], 2.)
        saved = json.loads((self.root / 'outputs/ddpo/evals/t2i_compbench/default/distribution_summary.json').read_text())
        self.assertEqual(saved['summary'], result['ddpo'])


if __name__ == '__main__':
    unittest.main()
