import unittest

import torch
from torch import nn

from clover.utils.diversity_score import (
    frechet_inception_distance,
    inception_score,
    rollout_fid_scores,
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


if __name__ == "__main__":
    unittest.main()
