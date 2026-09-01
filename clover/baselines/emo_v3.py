"""Train the third-generation entropy-maximizing optimization baseline.

EMO v3 uses the corrected per-timestep reward normalization, scheduled
reward-vs-entropy weight, capped new/old log-probability quotient, and decaying
learning rate implemented by clover.baselines.emo_v2. It keeps a separate
configuration type and artifact namespace so experiments can run and be
compared independently through main.py or the emo_v3 module entry point.
"""

from __future__ import annotations

from dataclasses import dataclass

from clover.baselines.common import parse_config
from clover.baselines.emo_v2 import (
    EMOV2V2Config,
    capped_log_probability_ratio,
    learning_rate_for_epoch,
    normalize_rewards_per_timestep,
    reward_scale_for_epoch,
    sac_update,
    train as train_emo,
)


@dataclass
class EMOV3Config(EMOV2V2Config):
    """Configuration for an independently persisted EMO v3 experiment."""

    output_dir: str = "outputs/emo_v3"


def train(config: EMOV3Config) -> list[dict[str, float]]:
    """Train EMO v3 while writing only to the EMO v3 artifact namespace."""
    return train_emo(config, baseline_name="emo_v3")


def main() -> None:
    """Parse command-line configuration and train EMO v3."""
    train(parse_config(EMOV3Config, __doc__ or "Train EMO v3"))


if __name__ == "__main__":
    main()
