from __future__ import annotations

import math
from typing import TYPE_CHECKING

from bi_rbot.base import observations as observ

if TYPE_CHECKING:
    from bi_rbot.tasks.locomotion.locomotion_env import LocomotionEnv


def track_height(env: LocomotionEnv) -> float:
    return math.exp(
        -(((observ.get_body_height(env) - env.cmd[0]) / env.cfg.height_sigma) ** 2)
    )


def track_speed(env: LocomotionEnv) -> float:
    return math.exp(
        -(((observ.get_horizontal_speed(env) - env.cmd[1]) / env.cfg.speed_sigma) ** 2)
    )


def track_yaw_rate(env: LocomotionEnv) -> float:
    return math.exp(
        -(((observ.get_gyro(env)[2] - env.cmd[2]) / env.cfg.yaw_sigma) ** 2)
    )
