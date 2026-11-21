from __future__ import annotations

from dataclasses import dataclass, field

from bi_rbot.base import rewards
from bi_rbot.base.robot_config import RobotConfig
from bi_rbot.tasks.locomotion import rewards as locomotion_rewards


@dataclass
class LocomotionConfig(RobotConfig):
    force_symmetric: bool = True  # TEMP symmetry test

    height_range: tuple = (0.25, 0.42)
    speed_range: tuple = (-0.0, 0.0)
    yaw_rate_range: tuple = (0.0, 0.0)

    height_sigma: float = 0.1
    speed_sigma: float = 0.2
    yaw_sigma: float = 0.6

    resample_every: int = 500
    reward_terms: dict = field(
        default_factory=lambda: {
            "upright": (rewards.track_upright, 1.0),
            "action_rate": (rewards.track_action_rate, -0.1),
            "knee_opposite": (rewards.track_knee_opposite, -0.5),
            "hip_comparison": (rewards.track_hip_comparison, 0.6),
            "track_height": (locomotion_rewards.track_height, 2.5),
            "track_speed": (locomotion_rewards.track_speed, 1.5),
            "track_yaw_rate": (locomotion_rewards.track_yaw_rate, 1.0),
        }
    )
