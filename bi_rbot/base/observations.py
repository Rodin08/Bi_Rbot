from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

if TYPE_CHECKING:
    from bi_rbot.base.robot_env import RobotEnv


def get_body_up_axis(env: RobotEnv) -> NDArray:
    return env.data.xmat[env.body_id][6:9]


def get_gyro(env: RobotEnv) -> NDArray:
    return env.data.sensor("gyro").data


def get_accel(env: RobotEnv) -> NDArray:
    return env.data.sensor("accel").data / 9.81


def get_leg_joint_pos(env: RobotEnv) -> NDArray:
    return env.data.qpos[env.leg_qpos]


def get_leg_joint_vel(env: RobotEnv) -> NDArray:
    return env.data.qvel[env.leg_qvel]


def get_wheel_vel(env: RobotEnv) -> NDArray:
    return env.data.qvel[env.wheel_qvel]


def get_body_height(env: RobotEnv) -> float:
    return env.data.xpos[env.body_id][2] - env.cfg.ground_z
