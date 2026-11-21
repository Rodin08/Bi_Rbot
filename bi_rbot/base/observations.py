from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray

from bi_rbot.base.kinematics import JointAngles, LegTarget, leg_fk

if TYPE_CHECKING:
    from bi_rbot.base.robot_env import RobotEnv


class CommonDiff(NamedTuple):
    common: float
    diff: float


# --- Internal helper functions ---
def _get_leg_joint_pos(env: RobotEnv) -> NDArray:
    return env.data.qpos[env.leg_qpos]


def _get_leg_joint_vel(env: RobotEnv) -> NDArray:
    return env.data.qvel[env.leg_qvel]


def _get_wheel_joint_vel(env: RobotEnv) -> NDArray:
    return env.data.qvel[env.wheel_qvel]


def _get_leg_task_space(env: RobotEnv) -> tuple[LegTarget, LegTarget]:
    hip_r, knee_r, hip_l, knee_l = _get_leg_joint_pos(env)
    right = leg_fk(JointAngles(hip_r, knee_r), env.cfg)
    left = leg_fk(JointAngles(hip_l, knee_l), env.cfg)
    return right, left


# --- Policy can access ---
def get_projected_up(env: RobotEnv) -> NDArray:
    return env.data.body(env.body_id).xmat[6:9]


def get_gyro(env: RobotEnv) -> NDArray:
    return env.data.sensor("gyro").data


def get_wheel_vel(env: RobotEnv) -> CommonDiff:
    right, left = _get_wheel_joint_vel(env)
    common = (left + right) / 2
    diff = (right - left) / 2
    return CommonDiff(common, diff)


def get_leg_length(env: RobotEnv) -> CommonDiff:
    right, left = _get_leg_task_space(env)
    return CommonDiff(
        (right.length + left.length) / 2, (right.length - left.length) / 2
    )


def get_leg_length_vel(env: RobotEnv) -> CommonDiff:
    t, s = env.cfg.thigh_length, env.cfg.shin_length
    _, knee_r, _, knee_l = _get_leg_joint_pos(env)
    _, knee_vel_r, _, knee_vel_l = _get_leg_joint_vel(env)
    right, left = _get_leg_task_space(env)

    right_vel = -t * s * math.sin(knee_r) * knee_vel_r / right.length
    left_vel = -t * s * math.sin(knee_l) * knee_vel_l / left.length
    return CommonDiff((right_vel + left_vel) / 2, (right_vel - left_vel) / 2)


def get_leg_angle(env: RobotEnv) -> CommonDiff:
    right, left = _get_leg_task_space(env)
    return CommonDiff((right.angle + left.angle) / 2, (right.angle - left.angle) / 2)


def get_leg_angle_vel(env: RobotEnv) -> CommonDiff:
    t, s = env.cfg.thigh_length, env.cfg.shin_length
    _, knee_r, _, knee_l = _get_leg_joint_pos(env)
    hip_vel_r, knee_vel_r, hip_vel_l, knee_vel_l = _get_leg_joint_vel(env)
    right, left = _get_leg_task_space(env)

    right_vel = hip_vel_r + s * (s + t * math.cos(knee_r)) / right.length**2 * knee_vel_r
    left_vel = hip_vel_l + s * (s + t * math.cos(knee_l)) / left.length**2 * knee_vel_l
    return CommonDiff((right_vel + left_vel) / 2, (right_vel - left_vel) / 2)


# --- Named components ---
def get_roll_rate(env: RobotEnv) -> float:
    return float(get_gyro(env)[1])


def get_pitch_rate(env: RobotEnv) -> float:
    return float(get_gyro(env)[0])


def get_yaw_rate(env: RobotEnv) -> float:
    return float(get_gyro(env)[2])


# --- Policy cannot access ---
def get_forward_vel(env: RobotEnv) -> float:
    forward = env.data.body(env.body_id).xmat.reshape(3, 3)[:, 1].copy()
    forward[2] = 0
    forward /= np.linalg.norm(forward)
    return float(env.data.qvel[:3] @ forward)


def get_body_height(env: RobotEnv) -> float:
    return float(env.data.body(env.body_id).xpos[2] - env.cfg.ground_z)
