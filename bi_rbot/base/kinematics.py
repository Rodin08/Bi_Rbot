from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import math

if TYPE_CHECKING:
    from bi_rbot.base.robot_config import RobotConfig


class LegTarget(NamedTuple):
    length: float
    angle: float


class JointAngles(NamedTuple):
    hip: float
    knee: float


def leg_ik(target: LegTarget, cfg: RobotConfig) -> JointAngles:
    r, theta = target.length, target.angle
    t, s = cfg.thigh_length, cfg.shin_length

    cos_knee = (r**2 - t**2 - s**2) / (2 * t * s)
    knee = cfg.knee_sign * math.acos(min(max(cos_knee, -1.0), 1.0))

    hip = theta - math.atan2(s * math.sin(knee), t + s * math.cos(knee))
    return JointAngles(hip=hip, knee=knee)


def leg_fk(joints: JointAngles, cfg: RobotConfig) -> LegTarget:
    hip, knee = joints.hip, joints.knee
    t, s = cfg.thigh_length, cfg.shin_length

    length = math.sqrt(t**2 + s**2 + 2 * t * s * math.cos(knee))
    angle = hip + math.atan2(s * math.sin(knee), t + s * math.cos(knee))
    return LegTarget(length=length, angle=angle)
