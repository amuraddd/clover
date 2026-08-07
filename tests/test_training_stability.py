import inspect
import unittest

import torch
from diffusers import DDPMScheduler

from clover.baselines.b2diffurl import B2DiffuRLConfig
from clover.baselines.ddpo import DDPOConfig
from clover.baselines.dpok import DPOKConfig
from clover.baselines.md3po import MD3POConfig
from clover.utils.baseline_utils import (
    ddpm_mean_std,
    ddpm_step_with_log_prob,
    is_stochastic_ddpm_transition,
    ppo_update,
)
from main import build_default_argv


class TrainingStabilityTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
        )
        self.scheduler.set_timesteps(50)

    def test_terminal_transition_is_sampled_but_not_trainable(self):
        timesteps = [int(timestep) for timestep in self.scheduler.timesteps]
        valid = [
            timestep
            for timestep in timesteps
            if is_stochastic_ddpm_transition(self.scheduler, timestep, min_std=1e-4)
        ]
        terminal = timesteps[-1]
        self.assertIn(terminal, (0, 1))
        self.assertEqual(len(valid), 49)
        self.assertNotIn(terminal, valid)

        sample = torch.zeros(2, 4, 2, 2)
        mean, std = ddpm_mean_std(self.scheduler, sample, terminal, sample)
        self.assertEqual(float(std), 0.0)
        next_sample, log_prob = ddpm_step_with_log_prob(
            self.scheduler, sample, terminal, sample
        )
        torch.testing.assert_close(next_sample, mean)
        torch.testing.assert_close(log_prob, torch.zeros(2))

    def test_ppo_uses_unclamped_log_ratio(self):
        source = inspect.getsource(ppo_update)
        self.assertIn("ratio = torch.exp(log_ratio)", source)
        self.assertNotIn("bounded_log_ratio", source)
        self.assertIn('"timestep_kl"', source)
        self.assertIn('"parameter_update_norm_mean"', source)

    def test_stronger_defaults_are_consistent(self):
        for config_type in (DDPOConfig, DPOKConfig, B2DiffuRLConfig, MD3POConfig):
            config = config_type()
            self.assertEqual(config.reward_type, "clip")
            self.assertEqual(config.learning_rate, 1e-5)
            self.assertEqual(config.lora_rank, 16)
            self.assertEqual(config.lora_alpha, 16)
            self.assertEqual(config.max_grad_norm, 1.0)
            self.assertEqual(config.min_log_prob_std, 1e-4)

    def test_main_uses_dpok_specific_update_epoch_flag(self):
        dpok_args = build_default_argv("clover.baselines.dpok")
        ddpo_args = build_default_argv("clover.baselines.ddpo")
        self.assertIn("--dpok-epochs", dpok_args)
        self.assertNotIn("--ppo-epochs", dpok_args)
        self.assertIn("--ppo-epochs", ddpo_args)


if __name__ == "__main__":
    unittest.main()
