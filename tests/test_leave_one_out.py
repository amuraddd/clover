import unittest

import torch

from clover.baselines.common import discount_rewards, leave_one_out_advantages


class LeaveOneOutTests(unittest.TestCase):
    def test_excludes_self_and_other_prompts(self):
        rewards = torch.tensor([1., 3., 8., 100.])
        result = leave_one_out_advantages(rewards, ['a', 'a', 'a', 'b'])
        torch.testing.assert_close(result, torch.tensor([-4.5, -1.5, 6., 100.]))
        changed = rewards.clone()
        changed[0] = 11.
        updated = leave_one_out_advantages(changed, ['a', 'a', 'a', 'b'])
        self.assertEqual(float(updated[0] - result[0]), 10.)

    def test_replay_never_contributes_to_baseline(self):
        result = leave_one_out_advantages(
            torch.tensor([2., 6., 100., 9.]), ['a', 'a', 'a', 'b'], reference_count=2,
        )
        torch.testing.assert_close(result, torch.tensor([-4., 4., 96., 9.]))

    def test_subtraction_precedes_discounting_without_normalization(self):
        result = discount_rewards(leave_one_out_advantages(
            torch.tensor([2., 6.]), ['a', 'a']), 3, .5)
        torch.testing.assert_close(result, torch.tensor([[-1., -2., -4.], [1., 2., 4.]]))

    def test_singleton_and_no_reference_fallback(self):
        rewards = torch.tensor([3.], dtype=torch.float64, requires_grad=True)
        for reference_count in (None, 0):
            result = leave_one_out_advantages(rewards, ['a'], reference_count)
            torch.testing.assert_close(result, rewards)
            self.assertFalse(result.requires_grad)
        result = leave_one_out_advantages(torch.empty(0), [])
        self.assertEqual(result.shape, (0,))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            leave_one_out_advantages(torch.ones(2), ['a'])
        with self.assertRaises(ValueError):
            leave_one_out_advantages(torch.ones(2), ['a', 'a'], 3)


if __name__ == '__main__':
    unittest.main()
