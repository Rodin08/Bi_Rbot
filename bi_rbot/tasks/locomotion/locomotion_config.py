from __future__ import annotations

from dataclasses import dataclass, field

from bi_rbot.base import rewards
from bi_rbot.base.robot_config import RobotConfig
from bi_rbot.tasks.locomotion.rewards import track_height, track_speed, track_yaw_rate


@dataclass
class LocomotionConfig(RobotConfig):
    height_range: tuple = (0.25, 0.42)
    speed_range: tuple = (-0.0, 0.0)
    yaw_rate_range: tuple = (0.0, 0.0)

    height_sigma: float = 0.05
    speed_sigma: float = 0.1
    yaw_sigma: float = 0.3

    resample_every: int = 500
    reward_terms: dict = field(
        default_factory=lambda: {
            "upright": (rewards.upright, 2.0),
            "knee_opposite": (rewards.knee_opposite, -0.5),
            "track_height": (track_height, 2.0),
            "track_speed": (track_speed, 2.0),
            "track_yaw_rate": (track_yaw_rate, 1.0),
            "action_rate": (rewards.action_rate, -0.05),
        }
    )
