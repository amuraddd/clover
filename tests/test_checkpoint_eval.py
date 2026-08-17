import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import torch
from PIL import Image

from clover.utils.checkpoint_eval import (
    prompt_baseline_fid_scores,
    sample_baseline_checkpoints,
)


class CheckpointEvalTests(unittest.TestCase):
    @patch("clover.utils.checkpoint_eval.Path.is_file", return_value=True)
    @patch("clover.utils.checkpoint_eval._suppress_huggingface_output")
    @patch("clover.utils.checkpoint_eval.set_peft_model_state_dict")
    @patch("clover.utils.checkpoint_eval.torch.load", return_value={"lora_state_dict": {}})
    @patch("clover.utils.checkpoint_eval.get_prompts", return_value=((), ("prompt one", "prompt two")))
    @patch("clover.utils.checkpoint_eval._baseline_config")
    @patch("clover.utils.checkpoint_eval.load_lora_pipeline")
    def test_samples_each_prompt_with_each_seed(
        self,
        load_pipeline,
        baseline_config,
        _get_prompts,
        _torch_load,
        _set_state,
        suppress_output,
        _is_file,
    ):
        baseline_config.return_value = SimpleNamespace(
            negative_prompt="", height=8, width=8, num_inference_steps=2, guidance_scale=1.0
        )
        pipe = MagicMock()
        pipe.unet = MagicMock()
        pipe.return_value = SimpleNamespace(
            images=[Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
        )
        load_pipeline.return_value = pipe

        result = sample_baseline_checkpoints(
            ["ddpo"], 2, base_seed=50, device="cpu", batch_size=2
        )

        self.assertEqual(result.columns.tolist(), ["seed", "prompt", "image", "baseline"])
        self.assertEqual(len(result), 4)
        self.assertEqual(result["seed"].tolist(), [50, 51, 50, 51])
        self.assertEqual(result["prompt"].tolist(), ["prompt one"] * 2 + ["prompt two"] * 2)
        self.assertEqual(result["baseline"].unique().tolist(), ["ddpo"])
        self.assertTrue(all(isinstance(image, Image.Image) for image in result["image"]))
        suppress_output.assert_called_once_with()
        pipe.set_progress_bar_config.assert_called_once_with(disable=True)

    def test_rejects_nonpositive_sample_count(self):
        with self.assertRaisesRegex(ValueError, "num_images_per_prompt"):
            sample_baseline_checkpoints(["ddpo"], 0)

    @patch("clover.utils.checkpoint_eval._inception_outputs")
    def test_prompt_baseline_fid_scores_average_pairwise_distances(self, inception_outputs):
        dataframe = pd.DataFrame(
            {
                "prompt": ["one", "one", "one", "two", "two"],
                "image": [Image.new("RGB", (4, 4)) for _ in range(5)],
                "baseline": ["ddpo"] * 5,
            }
        )
        inception_outputs.side_effect = [
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
        ]

        result = prompt_baseline_fid_scores(dataframe, device="cpu")

        self.assertEqual(result.columns.tolist(), ["prompt", "FID score", "baseline"])
        self.assertEqual(result["prompt"].tolist(), ["one", "two"])
        self.assertAlmostEqual(result.loc[0, "FID score"], 4.0 / 3.0)
        self.assertAlmostEqual(result.loc[1, "FID score"], 4.0)
        self.assertEqual(inception_outputs.call_count, 2)


if __name__ == "__main__":
    unittest.main()
