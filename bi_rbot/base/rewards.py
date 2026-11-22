from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from bi_rbot.base import observations as observ
from bi_rbot.base.actions import Actions

if TYPE_CHECKING:
    from bi_rbot.base.robot_env import RobotEnv


# --- Building blocks ---
def gaussian_kernel(error: float, sigma: float) -> float:
    return math.exp(-((error / sigma) ** 2))


def laplace_kernel(error: float, sigma: float) -> float:
    return math.exp(-abs(error) / sigma)


def cauchy_kernel(error: float, sigma: float) -> float:
    return 1 / (1 + (error / sigma) ** 2)


def squared_penalty(value) -> float:
    return float(np.sum(np.square(value)))


def abs_penalty(value) -> float:
    return float(np.sum(np.abs(value)))


# --- Rewards ---
def reward_upright(env: RobotEnv) -> float:
    up = observ.get_projected_up(env)
    return gaussian_kernel(math.hypot(up[0], up[1]), env.cfg.upright_sigma)


# --- Penalties ---
def penalty_action_rate(env: RobotEnv) -> float:
    return squared_penalty(env.action - env.prev_action)


def penalty_leg_length_diff(env: RobotEnv) -> float:
    return squared_penalty(env.action[Actions.LEG_LENGTH_DIFF])
