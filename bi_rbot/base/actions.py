from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, NamedTuple

from bi_rbot.base.kinematics import LegTarget

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from bi_rbot.base.robot_config import RobotConfig


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


class Actions(IntEnum):
    FWD = 0
    TURN = auto()
    LEG_LENGTH = auto()
    LEG_LENGTH_DIFF = auto()
    LEG_ANGLE = auto()


class DecodedActions(NamedTuple):
    leg_l: LegTarget
    leg_r: LegTarget
    wheel_l: float
    wheel_r: float


def decode_actions(action: NDArray, cfg: RobotConfig) -> DecodedActions:
    low, high = cfg.leg_length_range
    leg_length = (low + high) / 2 + action[Actions.LEG_LENGTH] * (high - low) / 2

    leg_length_diff = action[Actions.LEG_LENGTH_DIFF] * cfg.leg_length_diff_max

    leg_angle = action[Actions.LEG_ANGLE] * cfg.leg_angle_max

    wheel_l = _clamp(action[Actions.FWD] - action[Actions.TURN], -1, 1)
    wheel_r = _clamp(action[Actions.FWD] + action[Actions.TURN], -1, 1)

    return DecodedActions(
        leg_l=LegTarget(_clamp(leg_length - leg_length_diff, low, high), leg_angle),
        leg_r=LegTarget(_clamp(leg_length + leg_length_diff, low, high), leg_angle),
        wheel_l=wheel_l,
        wheel_r=wheel_r,
    )
