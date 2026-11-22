from __future__ import annotations

from dataclasses import dataclass, field

from bi_rbot.base import rewards
from bi_rbot.base.robot_config import RobotConfig
from bi_rbot.tasks.locomotion import rewards as locomotion_rewards


@dataclass
class LocomotionConfig(RobotConfig):
    # --- Commands ---
    height_range: tuple[float, float] = (0.25, 0.42)
    forward_vel_range: tuple[float, float] = (-0.0, 0.0)
    zero_forward_vel_prob: float = 0.3
    yaw_rate_range: tuple[float, float] = (-0.0, 0.0)
    zero_yaw_rate_prob: float = 0.5
    resample_every: int = 500

    # --- Rewards ---
    height_sigma: float = 0.1
    forward_vel_sigma: float = 0.2
    yaw_rate_sigma: float = 0.6
    reward_terms: dict = field(
        default_factory=lambda: {
            # --- Positive ---
            "upright": (rewards.reward_upright, 1.0),
            "height": (locomotion_rewards.reward_height, 2.5),
            "forward_vel": (locomotion_rewards.reward_forward_vel, 1.5),
            "yaw_rate": (locomotion_rewards.reward_yaw_rate, 1.0),
            # --- Negative ---
            "action_rate": (rewards.penalty_action_rate, -0.1),
            "leg_length_diff": (rewards.penalty_leg_length_diff, -0.2),
        }
    )
