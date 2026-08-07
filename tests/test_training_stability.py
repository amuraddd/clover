import inspect
import unittest
from types import SimpleNamespace

import torch
from diffusers import DDPMScheduler

from clover.baselines.b2diffurl import B2DiffuRLConfig
from clover.baselines.ddpo import DDPOConfig
from clover.baselines.dpok import DPOKConfig
from clover.baselines.md3po import MD3POConfig
from clover.baselines.common import parse_config
from clover.utils.baseline_utils import (
    ddpm_mean_std,
    ddpm_step_with_log_prob,
    is_stochastic_ddpm_transition,
    ppo_update,
    predict_noise_chunked,
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

    def test_chunked_cfg_preserves_order_and_bounds_unet_batch(self):
        class RecordingUnet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []

            def forward(self, latents, timestep, encoder_hidden_states, return_dict=False):
                self.batch_sizes.append(latents.shape[0])
                return (latents,)

        pipe = SimpleNamespace(unet=RecordingUnet())
        latents = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1, 1).requires_grad_()
        prompt_embeds = torch.zeros(10, 1, 1)
        prediction = predict_noise_chunked(
            pipe, latents, 10, prompt_embeds, guidance_scale=5.0, chunk_size=2
        )
        torch.testing.assert_close(prediction, latents)
        self.assertEqual(pipe.unet.batch_sizes, [4, 4, 2])
        prediction.sum().backward()
        torch.testing.assert_close(latents.grad, torch.ones_like(latents))

    def test_ppo_uses_unclamped_log_ratio(self):
        source = inspect.getsource(ppo_update)
        self.assertIn("ratio = torch.exp(log_ratio)", source)
        self.assertNotIn("bounded_log_ratio", source)
        self.assertIn('"timestep_kl"', source)
        self.assertIn('"parameter_update_norm_mean"', source)
        self.assertIn("predict_noise_chunked", source)

    def test_stronger_defaults_are_consistent(self):
        for config_type in (DDPOConfig, DPOKConfig, B2DiffuRLConfig, MD3POConfig):
            config = config_type()
            self.assertEqual(config.reward_type, "clip")
            self.assertEqual(config.learning_rate, 1e-5)
            self.assertEqual(config.lora_rank, 16)
            self.assertEqual(config.lora_alpha, 16)
            self.assertEqual(config.max_grad_norm, 1.0)
            self.assertEqual(config.min_log_prob_std, 1e-4)
            self.assertEqual(config.rollout_chunk_size, 32)

    def test_generated_arguments_parse_for_every_baseline(self):
        cases = (
            ("clover.baselines.md3po", MD3POConfig, "ppo_epochs"),
            ("clover.baselines.ddpo", DDPOConfig, "ppo_epochs"),
            ("clover.baselines.dpok", DPOKConfig, "dpok_epochs"),
            ("clover.baselines.b2diffurl", B2DiffuRLConfig, "ppo_epochs"),
        )
        for module, config_type, epoch_field in cases:
            generated = build_default_argv(module)
            config = parse_config(config_type, module, generated[1:])
            self.assertEqual(getattr(config, epoch_field), 2)
            self.assertEqual(config.lora_alpha, 16)
            self.assertEqual(config.min_log_prob_std, 1e-4)
            self.assertEqual(config.rollout_chunk_size, 32)
            self.assertEqual(config.reward_type, "clip")

    def test_main_uses_dpok_specific_update_epoch_flag(self):
        dpok_args = build_default_argv("clover.baselines.dpok")
        ddpo_args = build_default_argv("clover.baselines.ddpo")
        self.assertIn("--dpok-epochs", dpok_args)
        self.assertNotIn("--ppo-epochs", dpok_args)
        self.assertIn("--ppo-epochs", ddpo_args)


if __name__ == "__main__":
    unittest.main()
