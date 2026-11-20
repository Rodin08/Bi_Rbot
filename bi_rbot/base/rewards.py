from __future__ import annotations

import math
from typing import TYPE_CHECKING

from bi_rbot.base import observations as observ

if TYPE_CHECKING:
    from bi_rbot.base.robot_env import RobotEnv


def track_upright(env: RobotEnv) -> float:
    up = observ.get_body_up_axis(env)
    tilt = up[0] ** 2 + up[1] ** 2  # roll² + pitch², same weight for both
    return math.exp(-tilt / env.cfg.upright_sigma**2)


def track_action_rate(env: RobotEnv) -> float:
    return float(sum((env.action - env.prev_action) ** 2))


def track_alive(env: RobotEnv) -> float:
    return env.cfg.alive_reward


def track_knee_opposite(env: RobotEnv) -> float:
    knee = observ.get_leg_joint_pos(env)
    return max(0.0, -knee[1] * knee[3])


def track_hip_comparison(env: RobotEnv) -> float:
    hip = observ.get_leg_joint_pos(env)
    return math.exp(-(((hip[0] - hip[2]) / env.cfg.hip_comparison_flexibility) ** 2))
