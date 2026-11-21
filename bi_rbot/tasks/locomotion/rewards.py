from __future__ import annotations

from typing import TYPE_CHECKING

from bi_rbot.base import observations as observ
from bi_rbot.base.rewards import gaussian_kernel

if TYPE_CHECKING:
    from bi_rbot.tasks.locomotion.locomotion_env import LocomotionEnv


# --- Rewards ---
def reward_height(env: LocomotionEnv) -> float:
    error = observ.get_body_height(env) - env.cmd[0]
    return gaussian_kernel(error, env.cfg.height_sigma)


def reward_forward_vel(env: LocomotionEnv) -> float:
    error = observ.get_forward_vel(env) - env.cmd[1]
    return gaussian_kernel(error, env.cfg.speed_sigma)


def reward_yaw_rate(env: LocomotionEnv) -> float:
    error = observ.get_yaw_rate(env) - env.cmd[2]
    return gaussian_kernel(error, env.cfg.yaw_sigma)
