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
        -(
            (
                (observ.get_body_height(env) - env.cfg.target_height)
                / env.cfg.target_height_flexibility
            )
            ** 2
        )
    )


def action_rate(env: RobotEnv) -> float:
    return float(sum((env.action - env.prev_action) ** 2))


def alive(env: RobotEnv) -> float:
    return env.cfg.alive_reward


def knee_opposite(env: RobotEnv) -> float:
    knee = observ.get_leg_joint_pos(env)
    return max(0.0, -knee[1] * knee[3])


def hip_comparison(env: RobotEnv) -> float:
    hip = observ.get_leg_joint_pos(env)
    return math.exp(-(((hip[0] - hip[2]) / env.cfg.hip_comparison_flexibility) ** 2))
