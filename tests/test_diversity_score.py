import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import torch
from PIL import Image
from torch import nn

from clover.utils.diversity_score import (
    frechet_inception_distance,
    inception_score,
    rollout_fid_scores,
    successive_epoch_inception_metrics,
)


class MeanFeatureModel(nn.Module):
    def forward(self, images):
        return images.mean(dim=(2, 3))


class MeanClassifier(nn.Module):
    def forward(self, images):
        means = images.mean(dim=(1, 2, 3))
        return torch.stack((means, -means, means * 0.5), dim=1)


class DiversityScoreTests(unittest.TestCase):
    def setUp(self):
        self.images = torch.stack(
            (
                torch.zeros(3, 12, 12),
                torch.full((3, 12, 12), 0.25),
                torch.full((3, 12, 12), 0.75),
                torch.ones(3, 12, 12),
            )
        )

    def test_rollout_fid_scores_are_normalized_and_masked(self):
        mask = torch.tensor(((False, True), (True, False)))
        scores = rollout_fid_scores(
            self.images[:2], self.images[2:], mask, device="cpu", feature_model=MeanFeatureModel()
        )
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(torch.all((scores >= 0) & (scores <= 1)))
        self.assertLess(float(scores[0]), float(scores[1]))

    def test_inception_score_returns_finite_statistics(self):
        mean, std = inception_score(
            self.images, splits=2, device="cpu", classifier=MeanClassifier()
        )
        self.assertGreaterEqual(mean, 1.0)
        self.assertGreaterEqual(std, 0.0)

    def test_fid_is_zero_for_identical_sets(self):
        score = frechet_inception_distance(
            self.images, self.images.clone(), device="cpu", feature_model=MeanFeatureModel()
        )
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_fid_supports_singleton_epoch_groups(self):
        score = frechet_inception_distance(
            self.images[:1], self.images[1:2], device="cpu", feature_model=MeanFeatureModel()
        )
        self.assertGreater(score, 0.0)

    def test_successive_epoch_metrics_returns_requested_schema(self):
        with TemporaryDirectory() as directory:
            paths = []
            for epoch in (2, 4, 6):
                path = Path(directory) / f"epoch_{epoch}.png"
                Image.new("RGB", (4, 4), color=(epoch, epoch, epoch)).save(path)
                paths.append(path.name)
            dataframe = pd.DataFrame(
                {
                    "epoch": [2, 4, 6],
                    "image": paths,
                    "baseline": ["example"] * 3,
                    "prompt": ["prompt one", "prompt two", "prompt one"],
                }
            )
            with (
                patch("clover.utils.diversity_score.frechet_inception_distance", return_value=3.5),
                patch("clover.utils.diversity_score.inception_score", return_value=(1.25, 0.0)),
            ):
                result = successive_epoch_inception_metrics(dataframe, image_root=directory)

        self.assertEqual(
            result.columns.tolist(),
            ["Epoch", "Baseline", "prompt", "FID Score", "Inception Score"],
        )
        self.assertEqual(result["Epoch"].tolist(), ["4-2", "6-4"])
        self.assertEqual(result["FID Score"].tolist(), [3.5, 3.5])
        self.assertEqual(result["prompt"].tolist(), ["all prompts", "all prompts"])

    def test_successive_epoch_metrics_scores_all_prompts_at_each_epoch(self):
        dataframe = pd.DataFrame(
            {
                "epoch": [2, 2, 4, 4],
                "image": ["a.png", "b.png", "c.png", "d.png"],
                "baseline": ["example"] * 4,
                "prompt": ["prompt one", "prompt two"] * 2,
            }
        )
        opened = []

        def fake_open(path):
            opened.append(Path(path).name)
            return Image.new("RGB", (4, 4))

        with (
            patch("clover.utils.diversity_score.Image.open", side_effect=fake_open),
            patch("clover.utils.diversity_score.frechet_inception_distance", return_value=3.5),
            patch("clover.utils.diversity_score.inception_score", return_value=(1.25, 0.0)) as score,
        ):
            successive_epoch_inception_metrics(dataframe)

        self.assertEqual(opened, ["a.png", "b.png", "c.png", "d.png"])
        self.assertEqual(len(score.call_args.args[0]), 2)


if __name__ == "__main__":
    unittest.main()
