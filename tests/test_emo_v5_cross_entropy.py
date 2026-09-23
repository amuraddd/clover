"""CPU tests of cross-entropy math and update wiring without loading diffusion models."""
import ast
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[1] / 'clover/baselines/emo_v5.py'


def load_functions():
    tree = ast.parse(SOURCE.read_text())
    names = {'_diagonal_gaussian_cross_entropy', 'sac_update', 'capped_log_probability_ratio'}
    namespace = dict(torch=torch, Tensor=torch.Tensor, math=math, np=np,
                     deepcopy=deepcopy, SimpleNamespace=SimpleNamespace)
    exec(compile(ast.Module(body=[node for node in tree.body
                                 if isinstance(node, ast.FunctionDef) and node.name in names],
                            type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class EMOV5CrossEntropyTests(unittest.TestCase):
    def test_cross_entropy_matches_torch_and_only_current_parameters_get_gradients(self):
        cross_entropy_fn = load_functions()['_diagonal_gaussian_cross_entropy']
        mean = torch.tensor([[.3, -.5]], requires_grad=True)
        variance = torch.tensor([[.4, 2.]], requires_grad=True)
        ref_mean = torch.tensor([[0., .2]], requires_grad=True)
        ref_var = torch.tensor([[1., .8]], requires_grad=True)
        actual = cross_entropy_fn(mean, variance, ref_mean, ref_var)
        expected = torch.distributions.kl_divergence(
            torch.distributions.Normal(mean, variance.sqrt()),
            torch.distributions.Normal(ref_mean, ref_var.sqrt()),
        ).mean(1) + torch.distributions.Normal(mean, variance.sqrt()).entropy().mean(1)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertGreater(mean.grad.abs().sum().item(), 0)
        self.assertGreater(variance.grad.abs().sum().item(), 0)
        self.assertIsNone(ref_mean.grad)
        self.assertIsNone(ref_var.grad)
        torch.testing.assert_close(
            cross_entropy_fn(mean, variance, mean, variance),
            torch.distributions.Normal(mean, variance.sqrt()).entropy().mean(1),
        )
        with self.assertRaises(FloatingPointError):
            cross_entropy_fn(mean, torch.zeros_like(variance), ref_mean, ref_var)

    def test_config_defaults_validation_and_cli(self):
        import argparse
        from dataclasses import dataclass, field, fields
        namespace = dict(dataclass=dataclass, field=field, fields=fields, math=math,
                         B2_FULL_TRAIN_PROMPTS=(), B2_FULL_EVAL_PROMPTS=(),
                         argparse=argparse)
        tree = ast.parse(SOURCE.read_text())
        config_node = next(node for node in tree.body
                           if isinstance(node, ast.ClassDef) and node.name == 'EMOV2V2Config')
        common = SOURCE.with_name('common.py')
        parse_node = next(node for node in ast.parse(common.read_text()).body
                          if isinstance(node, ast.FunctionDef) and node.name == 'parse_config')
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
                            names=[ast.alias(name='annotations')], level=0),
                            config_node, parse_node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), namespace)
        config_type = namespace['EMOV2V2Config']
        config = config_type()
        self.assertEqual(config.buffer_reset, 5)
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                config_type(buffer_reset=invalid)
        self.assertEqual(config.entropy_scale, .05)
        self.assertEqual(config.cross_entropy_coefficient, .01)
        self.assertFalse(hasattr(config, 'reward_scale'))
        self.assertFalse(hasattr(config, 'kl_coefficient'))
        parsed = namespace['parse_config'](config_type, 'test',
                    ['--entropy-scale', '.2', '--cross-entropy-coefficient', '.3', '--buffer-reset', '3'])
        self.assertEqual(parsed.buffer_reset, 3)
        self.assertEqual(parsed.entropy_scale, .2)
        self.assertEqual(parsed.cross_entropy_coefficient, .3)
        for name in ('entropy_scale', 'cross_entropy_coefficient'):
            for invalid in (-1., float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    config_type(**{name: invalid})

    def run_update(self, coefficient, entropy_scale=.05):
        ns = load_functions()
        unet = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.constant_(unet.weight, .2)
        pipe = SimpleNamespace(unet=unet, scheduler=SimpleNamespace(
            config=SimpleNamespace(variance_type='learned_range')))
        refs = []

        def predict(p, state, timestep, embeds, guidance, chunk):
            if p.unet is not unet:
                refs.append((id(p.unet), p.unet.weight.detach().clone(),
                             p.unet.weight.requires_grad, p.unet.training))
            return p.unet.weight.reshape(1, 1).expand(state.shape[0], 2)

        ns.update(
            _predict_emo_v5_chunked=predict,
            encode_prompts=lambda *args: None,
            leave_one_out_advantages=lambda rewards, prompts, reference_count: rewards,
            sigmoid_discounting=lambda rewards, num_steps, k: rewards[:, None].expand(-1, num_steps),
            _emo_v5_step_with_log_prob=lambda scheduler, output, timestep, state, **kw:
                (kw['prev_sample'], output[:, 0]),
            ddpm_mean_std=lambda scheduler, output, timestep, state, **kw: (output[:, :1], None),
            _learned_range_variance=lambda scheduler, output, timestep, state: output[:, 1:].exp(),
            _learned_range_ddpm_entropy=lambda scheduler, output, state, timestep: output[:, 1],
            trainable_parameters=lambda model: list(model.parameters()),
        )
        ns['deepcopy'] = Mock(wraps=deepcopy)
        config = SimpleNamespace(cross_entropy_coefficient=coefficient, entropy_scale=entropy_scale, clip_range=1e-4,
                                 eta=.7, gamma=.9, sac_epochs=3, minibatch_size=1,
                                 negative_prompt='', guidance_scale=1., rollout_chunk_size=1,
                                 likelihood_scale=1., max_grad_norm=100.)
        rollout = dict(states=torch.ones(1, 2, 1), actions=torch.ones(1, 2, 1),
                       old_log_probs=torch.zeros(1, 2), timesteps=torch.tensor([2, 1]),
                       prompts=['test'], rewards=torch.ones(1, 2))
        metrics = ns['sac_update'](pipe, rollout, torch.optim.SGD(unet.parameters(), lr=.1),
                                   config, torch.device('cpu'), torch.float32)
        return metrics, refs, ns['deepcopy'], unet.weight.detach().item()

    def test_reference_stays_frozen_and_penalty_affects_update(self):
        metrics, refs, copy_mock, weight = self.run_update(2.)
        copy_mock.assert_called_once()
        self.assertEqual(len({item[0] for item in refs}), 1)
        for _, value, requires_grad, training in refs:
            torch.testing.assert_close(value, torch.tensor([[.2]]))
            self.assertFalse(requires_grad)
            self.assertFalse(training)
        self.assertGreater(metrics['cross_entropy_objective'], 0)
        self.assertAlmostEqual(metrics['cross_entropy_penalty'], 2 * metrics['cross_entropy_objective'])
        disabled, no_refs, no_copy, unpenalized_weight = self.run_update(0.)
        no_copy.assert_not_called()
        self.assertEqual(no_refs, [])
        self.assertEqual(disabled['cross_entropy_penalty'], 0.)
        self.assertLess(weight, unpenalized_weight)
        self.assertAlmostEqual(metrics['entropy_bonus'], .05 * metrics['entropy_objective'])
        self.assertAlmostEqual(
            metrics['loss'], -metrics['policy_objective'] - metrics['entropy_bonus']
            + metrics['cross_entropy_penalty'], places=6,
        )
        no_entropy, _, _, no_entropy_weight = self.run_update(0., entropy_scale=0.)
        self.assertEqual(no_entropy['entropy_bonus'], 0.)
        self.assertGreater(unpenalized_weight, no_entropy_weight)
        self.assertNotIn('reward_scale', metrics)
        self.assertNotIn('kl_coefficient', metrics)


if __name__ == '__main__':
    unittest.main()
