"""Checkpoint RNG compatibility without requiring CUDA hardware."""
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from clover.utils import baseline_utils as utils


class CheckpointRNGTests(unittest.TestCase):
    def check_restore(self, saved_count, visible_count, device_type="cuda", missing=False):
        saved_states = [torch.tensor([i], dtype=torch.uint8) for i in range(saved_count)]
        current_states = [torch.tensor([100 + i], dtype=torch.uint8) for i in range(visible_count)]
        initial_states = list(current_states)
        checkpoint = {
            "epoch": 15,
            "history": [{"epoch": 15}],
            "lora_state_dict": {},
            "optimizer_state_dict": {"state": {}},
            "scheduler_state_dict": {"last_epoch": 15},
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "generator_state": torch.Generator().manual_seed(123).get_state(),
        }
        if not missing:
            checkpoint["cuda_rng_state"] = saved_states

        def restore(states):
            for i, state in enumerate(states):
                # Reproduce the original IndexError when a saved GPU is absent.
                current_states[i] = state

        optimizer, scheduler, generator = Mock(), Mock(), Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint" / "checkpoint.pt"
            path.parent.mkdir()
            torch.save(checkpoint, path)
            with patch.object(utils, "set_peft_model_state_dict"), \
                 patch.object(torch.cuda, "device_count", return_value=visible_count), \
                 patch.object(torch.cuda, "set_rng_state_all", side_effect=restore) as setter:
                result = utils.load_training_checkpoint(
                    SimpleNamespace(unet=Mock()), optimizer, directory,
                    torch.device(device_type), generator, scheduler,
                )
        self.assertEqual(result, (15, [{"epoch": 15}]))
        optimizer.load_state_dict.assert_called_once_with(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict.assert_called_once_with(checkpoint["scheduler_state_dict"])
        self.assertTrue(torch.equal(generator.set_state.call_args.args[0], checkpoint["generator_state"]))
        if device_type == "cpu" or missing:
            setter.assert_not_called()
        for i, state in enumerate(current_states):
            expected = (saved_states[i] if device_type == "cuda" and not missing
                        and i < saved_count else initial_states[i])
            self.assertTrue(torch.equal(state, expected))

    def test_fewer_visible_gpus(self):
        self.check_restore(2, 1)

    def test_matching_gpu_counts(self):
        self.check_restore(2, 2)

    def test_additional_gpus_keep_initialized_states(self):
        self.check_restore(1, 2)

    def test_cpu_does_not_restore_cuda_states(self):
        self.check_restore(2, 0, device_type="cpu")

    def test_legacy_checkpoint_without_cuda_states(self):
        self.check_restore(0, 2, missing=True)

    def test_empty_cuda_states_keep_initialized_states(self):
        self.check_restore(0, 2)


if __name__ == "__main__":
    unittest.main()
