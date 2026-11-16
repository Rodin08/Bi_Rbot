import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RobotConfig:
    xml_path: Path = Path(__file__).parent.parent / "assets" / "bi_rbot_model.xml"

    frame_skip: int = 5
    max_steps: int = 1000

    action_scale: list = field(default_factory=lambda: [1.57, 1.57, 2.44, 2.44, 1, 1])

    ground_z: float = -0.5
    min_height: float = 0.15
    target_height: float = 0.3
    min_straight_z: float = math.sin(
        math.radians(45)
    )  # 90 would mean perfectly straight

    reward_terms: dict = field(default_factory=dict)
