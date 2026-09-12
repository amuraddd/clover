"""CPU tests of KL math and update wiring without loading diffusion models."""
import ast
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[1] / 'clover/baselines/emo_v4.py'


def load_functions():
    tree = ast.parse(SOURCE.read_text())
    names = {'_diagonal_gaussian_kl', 'sac_update', 'capped_log_probability_ratio'}
    namespace = dict(torch=torch, Tensor=torch.Tensor, math=math, np=np,
                     deepcopy=deepcopy, SimpleNamespace=SimpleNamespace)
    exec(compile(ast.Module(body=[node for node in tree.body
                                 if isinstance(node, ast.FunctionDef) and node.name in names],
                            type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class EMOV4KLTests(unittest.TestCase):
    def test_kl_matches_torch_and_only_current_parameters_get_gradients(self):
        kl_fn = load_functions()['_diagonal_gaussian_kl']
        mean = torch.tensor([[.3, -.5]], requires_grad=True)
        variance = torch.tensor([[.4, 2.]], requires_grad=True)
        ref_mean = torch.tensor([[0., .2]], requires_grad=True)
        ref_var = torch.tensor([[1., .8]], requires_grad=True)
        actual = kl_fn(mean, variance, ref_mean, ref_var)
        expected = torch.distributions.kl_divergence(
            torch.distributions.Normal(mean, variance.sqrt()),
            torch.distributions.Normal(ref_mean, ref_var.sqrt()),
        ).mean(1)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertGreater(mean.grad.abs().sum().item(), 0)
        self.assertGreater(variance.grad.abs().sum().item(), 0)
        self.assertIsNone(ref_mean.grad)
        self.assertIsNone(ref_var.grad)
        torch.testing.assert_close(kl_fn(mean, variance, mean, variance), torch.zeros(1))
        with self.assertRaises(FloatingPointError):
            kl_fn(mean, torch.zeros_like(variance), ref_mean, ref_var)

    def run_update(self, coefficient):
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
            _predict_emo_v4_chunked=predict,
            encode_prompts=lambda *args: None,
            leave_one_out_advantages=lambda rewards, prompts, reference_count: rewards,
            discount_rewards=lambda rewards, steps, gamma: rewards[:, None].expand(-1, steps),
            _emo_v4_step_with_log_prob=lambda scheduler, output, timestep, state, **kw:
                (kw['prev_sample'], output[:, 0]),
            ddpm_mean_std=lambda scheduler, output, timestep, state, **kw: (output[:, :1], None),
            _learned_range_variance=lambda scheduler, output, timestep, state: output[:, 1:].exp(),
            _learned_range_ddpm_entropy=lambda scheduler, output, state, timestep: output[:, 1],
            trainable_parameters=lambda model: list(model.parameters()),
        )
        ns['deepcopy'] = Mock(wraps=deepcopy)
        config = SimpleNamespace(kl_coefficient=coefficient, reward_scale=1., clip_range=1e-4,
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
        self.assertGreater(metrics['kl_mean'], 0)
        self.assertAlmostEqual(metrics['kl_penalty'], 2 * metrics['kl_mean'])
        disabled, no_refs, no_copy, unpenalized_weight = self.run_update(0.)
        no_copy.assert_not_called()
        self.assertEqual(no_refs, [])
        self.assertEqual(disabled['kl_penalty'], 0.)
        self.assertLess(weight, unpenalized_weight)


if __name__ == '__main__':
    unittest.main()
