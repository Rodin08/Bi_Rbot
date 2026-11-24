from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RobotConfig:
    # --- Simulation ---
    xml_path: Path = Path(__file__).parent.parent / "assets" / "bi_rbot_model.xml"
    frame_skip: int = 5
    max_steps: int = 1000
    ground_z: float = -0.5

    # --- Robot geometry ---
    thigh_length: float = 0.2
    shin_length: float = 0.2
    knee_sign: int = 1
    wheel_radius: float = 0.07

    # --- Actions ---
    action_smoothing: float = 0.4
    leg_length_range: tuple[float, float] = (0.15, 0.38)
    leg_length_diff_max: float = 0.05
    leg_angle_max: float = 0.5

    # --- Termination ---
    terminate_on_body_contact: bool = True
    min_height: float = 0.15
    max_tilt_deg: float = 40

    # --- Rewards ---
    upright_sigma: float = 0.2
    reward_terms: dict = field(default_factory=dict)
