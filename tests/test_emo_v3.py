import unittest
from unittest.mock import patch

import torch

from clover.baselines.emo_v2 import (
    EMOV2V2Config,
    capped_log_probability_ratio,
    emo_v2_combined_rollouts,
    learning_rate_for_epoch,
    normalize_rewards_per_timestep,
    reward_scale_for_epoch,
)
from clover.baselines.emo_v3 import EMOV3Config
from clover.baselines.common import parse_config
from main import BASELINES, build_default_argv


class EMOV3Tests(unittest.TestCase):
    def test_rewards_are_normalized_independently_per_timestep(self):
        rewards = torch.tensor(
            [
                [1.0, 10.0, 100.0],
                [3.0, 14.0, 104.0],
                [5.0, 18.0, 108.0],
            ]
        )

        normalized = normalize_rewards_per_timestep(rewards)

        torch.testing.assert_close(
            normalized.mean(dim=0),
            torch.zeros(3),
            atol=1e-7,
            rtol=0.0,
        )
        torch.testing.assert_close(
            normalized.std(dim=0, unbiased=False),
            torch.ones(3),
            atol=1e-7,
            rtol=0.0,
        )

    def test_reward_and_learning_rate_schedules_change_after_each_decade(self):
        self.assertEqual(EMOV2V2Config().reward_scale, 10.0)
        self.assertEqual(reward_scale_for_epoch(10.0, 1), 10.0)
        self.assertEqual(reward_scale_for_epoch(10.0, 10), 10.0)
        self.assertEqual(reward_scale_for_epoch(10.0, 11), 20.0)
        self.assertEqual(reward_scale_for_epoch(10.0, 21), 40.0)

        self.assertEqual(learning_rate_for_epoch(1e-4, 1), 1e-4)
        self.assertEqual(learning_rate_for_epoch(1e-4, 10), 1e-4)
        self.assertAlmostEqual(learning_rate_for_epoch(1e-4, 11), 1e-5)
        self.assertAlmostEqual(learning_rate_for_epoch(1e-4, 21), 1e-6)

    def test_log_probability_quotient_is_capped_and_handles_zero_old_value(self):
        new = torch.tensor([-3.0, -2.0, 7.0], requires_grad=True)
        old = torch.tensor([-2.0, -4.0, 0.0])

        ratio = capped_log_probability_ratio(new, old)

        torch.testing.assert_close(ratio, torch.tensor([1.0, 0.5, 1.0]))
        self.assertFalse(ratio.requires_grad)

    def test_replay_requires_diversity_and_same_prompt_current_mean_reward(self):
        def rollout(prompts, terminal_rewards):
            batch_size = len(prompts)
            trajectory_length = 2
            rewards = torch.zeros(batch_size, trajectory_length)
            rewards[:, -1] = torch.tensor(terminal_rewards)
            return {
                "prompts": prompts,
                "states": torch.zeros(batch_size, trajectory_length, 1),
                "actions": torch.zeros(batch_size, trajectory_length, 1),
                "old_log_probs": torch.zeros(batch_size, trajectory_length),
                "timesteps": torch.tensor([1, 0]),
                "rewards": rewards,
                "images": [f"image-{index}" for index in range(batch_size)],
            }

        current = rollout(
            ["a", "a", "b", "b"],
            [0.25, 0.75, 0.5, 1.0],
        )
        replay = rollout(
            ["a", "a", "b", "b", "a", "c"],
            [0.49, 0.5, 0.74, 0.75, 0.9, 0.99],
        )

        with patch(
            "clover.baselines.emo_v2.rollout_fid_scores",
            return_value=torch.tensor([0.5, 0.5, 0.5, 0.5, 0.1]),
        ):
            combined = emo_v2_combined_rollouts(
                current,
                trajectories={1: replay},
                diversity_threshold=0.3,
                required_trajectory_epoch=1,
            )

        self.assertEqual(combined["prompts"], ["a", "a", "b", "b", "a", "b"])
        torch.testing.assert_close(
            combined["rewards"][-2:, -1],
            torch.tensor([0.5, 0.75]),
        )

    def test_emo_v3_is_active_and_parses_main_arguments(self):
        self.assertIn(("emo_v3", "clover.baselines.emo_v3"), BASELINES)
        arguments = build_default_argv(
            "clover.baselines.emo_v3", seed=456, gpu_ids=[0]
        )
        config = parse_config(EMOV3Config, "emo_v3", arguments[1:])
        scale_flag = arguments.index("--reward-scale")
        self.assertEqual(arguments[scale_flag + 1], "10.0")

        self.assertEqual(config.output_dir, "outputs/emo_v3/seed_456")
        self.assertEqual(config.reward_scale, 10.0)
        self.assertEqual(config.gpu_ids, [0])
        self.assertEqual(config.sac_epochs, 4)


if __name__ == "__main__":
    unittest.main()
