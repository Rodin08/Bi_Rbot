from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from bi_rbot.tasks.locomotion.locomotion_config import LocomotionConfig


class Commands(IntEnum):
    HEIGHT = 0
    FORWARD_VEL = auto()
    YAW_RATE = auto()


def _sample_or_zero(
    rng: np.random.Generator, value_range: tuple[float, float], zero_prob: float
) -> float:
    if rng.random() >= zero_prob:
        return rng.uniform(*value_range)
    return 0.0


def sample(rng: np.random.Generator, cfg: LocomotionConfig) -> NDArray:
    cmd = np.zeros(len(Commands))
    cmd[Commands.HEIGHT] = rng.uniform(*cfg.height_range)
    cmd[Commands.FORWARD_VEL] = _sample_or_zero(
        rng, cfg.forward_vel_range, cfg.zero_forward_vel_prob
    )
    cmd[Commands.YAW_RATE] = _sample_or_zero(
        rng, cfg.yaw_rate_range, cfg.zero_yaw_rate_prob
    )
    return cmd
