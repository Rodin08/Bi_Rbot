from __future__ import annotations

import math
from typing import TYPE_CHECKING

from bi_rbot.base import observations as observ

if TYPE_CHECKING:
    from bi_rbot.base.robot_env import RobotEnv


def upright(env: RobotEnv) -> float:
    return observ.get_body_up_axis(env)[2]


def height(env: RobotEnv) -> float:
    return math.exp(
        -(((observ.get_body_height(env) - env.cfg.target_height) / 0.1) ** 2)
    )


def action_rate(env: RobotEnv) -> float:
    raise NotImplementedError  ### Temporary will be something like sum((action - prev_action) ** 2) ###


def alive(env: RobotEnv) -> float:
    return 1.0
