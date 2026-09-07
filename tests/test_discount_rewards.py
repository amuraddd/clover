import unittest

import torch

from clover.baselines.common import discount_rewards, normalize_rewards, sigmoid_discounting, normalize_rewards_per_trajectory


class DiscountRewardsTests(unittest.TestCase):
    def test_terminal_returns_and_input_preservation(self):
        rewards = torch.tensor([8., -4.], dtype=torch.float64, requires_grad=True)
        actual = discount_rewards(rewards, 4, 0.5)
        torch.testing.assert_close(actual, rewards.new_tensor([[1, 2, 4, 8], [-0.5, -1, -2, -4]]))
        self.assertEqual(actual.dtype, rewards.dtype)
        self.assertEqual(actual.device, rewards.device)
        actual.sum().backward()
        torch.testing.assert_close(rewards.grad, rewards.new_full((2,), 1.875))

    def test_normalize_then_discount_preserves_temporal_weight(self):
        terminal = torch.tensor([1., 3., 5.])
        normalized = normalize_rewards(terminal)
        result = discount_rewards(normalized, 3, 0.5)
        torch.testing.assert_close(result[:, -1], normalized)
        torch.testing.assert_close(result[:, 0], result[:, -1] * 0.25)
        torch.testing.assert_close(result.std(dim=0, unbiased=False), torch.tensor([0.25, 0.5, 1.]))

    def test_boundary_discounts(self):
        rewards = torch.tensor([2., 4.])
        torch.testing.assert_close(discount_rewards(rewards, 3, 0), torch.tensor([[0., 0., 2.], [0., 0., 4.]]))
        torch.testing.assert_close(discount_rewards(rewards, 3, 1), rewards[:, None].expand(-1, 3))
        torch.testing.assert_close(discount_rewards(rewards, 1, 0.5), rewards[:, None])

    def test_invalid_inputs(self):
        for steps, gamma in [(0, .5), (-1, .5), (1, -.1), (1, 1.1), (1, float('nan'))]:
            with self.subTest(steps=steps, gamma=gamma), self.assertRaises(ValueError):
                discount_rewards(torch.tensor([1.]), steps, gamma)
        with self.assertRaises(ValueError):
            discount_rewards(torch.ones(2, 3), 3, .5)


class SigmoidDiscountingTests(unittest.TestCase):
    def test_matches_sample_curve_without_mutating_input(self):
        terminal = torch.tensor([8., -4., 0.], dtype=torch.float64, requires_grad=True)
        x = torch.linspace(0, 1, 9, dtype=terminal.dtype)
        sigmoid = torch.sigmoid(5 * (x - 0.5))
        weights = (sigmoid - sigmoid[0]) / (sigmoid[-1] - sigmoid[0])
        result = sigmoid_discounting(terminal, 9)
        torch.testing.assert_close(result, terminal[:, None] * weights)
        torch.testing.assert_close(terminal, torch.tensor([8., -4., 0.], dtype=terminal.dtype))
        torch.testing.assert_close(result[:, 0], torch.zeros_like(terminal))
        torch.testing.assert_close(result[:, -1], terminal)
        result.sum().backward()
        torch.testing.assert_close(terminal.grad, torch.full_like(terminal, 4.5))

    def test_steepness_and_monotonic_weights(self):
        gentle = sigmoid_discounting(torch.ones(1), 9, k=1)[0]
        steep = sigmoid_discounting(torch.ones(1), 9, k=20)[0]
        self.assertTrue(bool((steep[1:] >= steep[:-1]).all()))
        self.assertLess(steep[2], gentle[2])
        self.assertGreater(steep[6], gentle[6])
        self.assertEqual(steep[4].item(), 0.5)

    def test_short_trajectories(self):
        terminal = torch.tensor([2., -3.])
        torch.testing.assert_close(sigmoid_discounting(terminal, 1), terminal[:, None])
        torch.testing.assert_close(sigmoid_discounting(terminal, 2), torch.tensor([[0., 2.], [0., -3.]]))

    def test_small_k_and_low_precision_remain_finite(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            terminal = torch.ones(1, dtype=dtype)
            result = sigmoid_discounting(terminal, 5, k=1e-12)
            self.assertEqual(result.dtype, dtype)
            self.assertEqual(result.device, terminal.device)
            torch.testing.assert_close(result[0], torch.linspace(0, 1, 5, dtype=dtype))

    def test_invalid_inputs(self):
        for steps, k in [(0, 5), (2, 0), (2, -1), (2, float('inf')), (2, float('nan'))]:
            with self.subTest(steps=steps, k=k), self.assertRaises(ValueError):
                sigmoid_discounting(torch.ones(1), steps, k)
        with self.assertRaises(ValueError):
            sigmoid_discounting(torch.ones(2, 3), 3)


class PerTrajectoryNormalizationTests(unittest.TestCase):
    def test_discount_then_normalize_each_row(self):
        discounted = discount_rewards(torch.tensor([2., 10.]), 4, 0.5)
        actual = normalize_rewards_per_trajectory(discounted)
        expected = torch.stack([normalize_rewards(row) for row in discounted])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.mean(dim=1), torch.zeros(2), atol=1e-6, rtol=0)
        torch.testing.assert_close(actual.std(dim=1, unbiased=False), torch.ones(2))
        self.assertTrue(bool((actual[:, 0] < 0).all()))
        self.assertTrue(bool((actual[:, -1] > 0).all()))

    def test_other_samples_do_not_affect_normalization(self):
        rewards = torch.tensor([[1., 2., 4.], [10., 30., 100.]])
        expected = normalize_rewards_per_trajectory(rewards)[0]
        rewards[1] = torch.tensor([-1000., 2000., 8000.])
        torch.testing.assert_close(normalize_rewards_per_trajectory(rewards)[0], expected)
        torch.testing.assert_close(normalize_rewards_per_trajectory(rewards[:1])[0], expected)

    def test_constant_and_single_step_rows_are_zero(self):
        for dtype in (torch.float16, torch.float32):
            for rewards in (torch.ones(2, 1, dtype=dtype), torch.ones(2, 4, dtype=dtype)):
                torch.testing.assert_close(normalize_rewards_per_trajectory(rewards), torch.zeros_like(rewards))


if __name__ == '__main__':
    unittest.main()
