"""CPU replay lifecycle checks without model loading."""
import ast
from pathlib import Path
import tempfile
import unittest

import torch

SOURCE = Path(__file__).resolve().parents[1] / 'clover/baselines/emo_v5.py'


def load_combiner():
    node = next(node for node in ast.parse(SOURCE.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == 'emo_v5_combined_rollouts')
    namespace = dict(torch=torch, Path=Path,
                     rollout_fid_scores=lambda reference, replay, **kwargs: torch.ones(len(replay)))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['emo_v5_combined_rollouts']


def rollout(epoch):
    return dict(epoch=epoch, states=torch.ones(1, 2, 1), actions=torch.ones(1, 2, 1),
                prompts=['test'], rewards=torch.ones(1, 2), timesteps=torch.tensor([2, 1]),
                old_log_probs=torch.zeros(1, 2), images=torch.ones(1, 3, 2, 2))


class ReplayResetTests(unittest.TestCase):
    def test_boundary_deletes_buffer_and_next_epoch_uses_new_data(self):
        combine = load_combiner()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trajectories.pt'
            archive = Path(directory) / 'history.json'
            archive.write_text('{}')
            torch.save({i: rollout(i) for i in range(1, 5)}, path)
            combined = combine(rollout(5), trajectory_path=path, required_trajectory_epoch=4)
            self.assertEqual(len(combined['prompts']), 5)
            torch.save({i: rollout(i) for i in range(1, 6)}, path)
            fresh = rollout(6)
            # Resume at epoch 6: infer the boundary from required previous epoch.
            self.assertIs(combine(fresh, trajectory_path=path, required_trajectory_epoch=5), fresh)
            self.assertFalse(path.exists())
            self.assertTrue(archive.exists())
            self.assertIs(combine(fresh, trajectory_path=path, epoch=6), fresh)
            # The saver restarts rollout keys at 1; recorded epoch stays authoritative.
            torch.save({1: rollout(6)}, path)
            combined = combine(rollout(7), trajectory_path=path, required_trajectory_epoch=6)
            self.assertEqual(len(combined['prompts']), 2)
            self.assertTrue(path.exists())
            with self.assertRaises(FileNotFoundError):
                combine(rollout(8), trajectory_path=Path(directory) / 'missing.pt', epoch=8,
                        required_trajectory_epoch=7)

    def test_in_memory_reset_and_custom_interval(self):
        combine = load_combiner()
        for epoch, interval in ((6, 5), (11, 5), (4, 3), (2, 1)):
            buffer = {1: rollout(1)}
            fresh = rollout(epoch)
            self.assertIs(combine(fresh, trajectories=buffer, epoch=epoch, buffer_reset=interval), fresh)
            self.assertEqual(buffer, {})
        for interval in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                combine(rollout(1), trajectories={}, buffer_reset=interval)


if __name__ == '__main__':
    unittest.main()
