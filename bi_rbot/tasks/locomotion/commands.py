import numpy as np

from bi_rbot.tasks.locomotion.locomotion_config import LocomotionConfig


def sample(rng: np.random.Generator, cfg: LocomotionConfig) -> np.ndarray:
    height = rng.uniform(cfg.height_range[0], cfg.height_range[1])
    speed = rng.uniform(cfg.speed_range[0], cfg.speed_range[1])
    yaw_rate = rng.uniform(cfg.yaw_rate_range[0], cfg.yaw_rate_range[1])
    return np.array([height, speed, yaw_rate])
