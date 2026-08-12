import inspect
import random
import unittest
from types import SimpleNamespace

import torch
from diffusers import DDPMScheduler

from clover.baselines import b2diffurl, ddpo, dpok, md3po
from clover.baselines.b2diffurl import B2DiffuRLConfig
from clover.baselines.ddpo import DDPOConfig
from clover.baselines.dpok import DPOKConfig
from clover.baselines.md3po import MD3POConfig
from clover.baselines.common import parse_config
from clover.utils.baseline_utils import (
    approximate_kl_from_log_ratio,
    ddpm_mean_std,
    ddpm_step_with_log_prob,
    is_stochastic_ddpm_transition,
    load_training_checkpoint,
    ppo_update,
    predict_noise_chunked,
    standard_eval_prompts,
    transition_log_prob,
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

    def test_likelihood_scale_scales_transition_log_prob(self):
        mean = torch.zeros(2, 4, 2, 2)
        std = torch.ones(())
        sample = torch.ones_like(mean)
        unscaled = transition_log_prob(mean, std, sample)
        scaled = transition_log_prob(mean, std, sample, likelihood_scale=1000.0)
        torch.testing.assert_close(scaled, unscaled * 1000.0)
        with self.assertRaisesRegex(ValueError, "likelihood_scale must be positive"):
            transition_log_prob(mean, std, sample, likelihood_scale=0.0)

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

    def test_approximate_kl_is_stable_for_tiny_log_ratios(self):
        log_ratio = torch.tensor([1e-5, -1e-5], dtype=torch.float32)
        approximate_kl = approximate_kl_from_log_ratio(log_ratio)

        self.assertEqual(approximate_kl.dtype, torch.float64)
        self.assertGreater(float(approximate_kl), 0.0)
        torch.testing.assert_close(
            approximate_kl,
            torch.tensor(5e-11, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-15,
        )

    def test_ppo_uses_unclamped_log_ratio(self):
        source = inspect.getsource(ppo_update)
        self.assertIn("ratio = torch.exp(log_ratio)", source)
        self.assertIn("approximate_kl_from_log_ratio(log_ratio)", source)
        self.assertNotIn("bounded_log_ratio", source)
        self.assertIn('"timestep_kl"', source)
        self.assertIn('"parameter_update_norm_mean"', source)
        self.assertIn("predict_noise_chunked", source)

    def test_md3po_replays_samples_above_diversity_threshold(self):
        source = inspect.getsource(md3po.md3po_combined_rollouts)
        self.assertIn("keep = fid_scores > diversity_threshold", source)
        self.assertNotIn("keep = fid_scores <= diversity_threshold", source)

    def test_each_baseline_uses_an_epoch_specific_rollout_seed(self):
        for module in (ddpo, dpok, b2diffurl, md3po):
            source = inspect.getsource(module.train)
            self.assertIn("generator = set_seed(config.seed + epoch, device)", source)

    def test_checkpoint_is_loaded_on_cpu_for_rng_state_restoration(self):
        source = inspect.getsource(load_training_checkpoint)
        self.assertIn('map_location="cpu"', source)
        self.assertNotIn("map_location=device", source)

    def test_standard_eval_prompts_do_not_reset_training_rng(self):
        config = SimpleNamespace(eval_prompts=tuple(f"prompt {index}" for index in range(20)))
        random.seed(987)
        expected = [random.random() for _ in range(3)]

        random.seed(987)
        first = random.random()
        prompts = standard_eval_prompts(config)
        remaining = [random.random() for _ in range(2)]

        self.assertEqual([first, *remaining], expected)
        self.assertEqual(prompts, standard_eval_prompts(config))

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
            self.assertGreater(config.likelihood_scale, 0)
        self.assertEqual(DDPOConfig().likelihood_scale, 1.0)
        for config_type in (DDPOConfig, DPOKConfig, B2DiffuRLConfig, MD3POConfig):
            self.assertEqual(config_type().clip_range, 1e-4)

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
