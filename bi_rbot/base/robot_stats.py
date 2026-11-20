from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bi_rbot.base.robot_env import RobotEnv


class EpisodeStats[EnvT: RobotEnv]:
    def __init__(self) -> None:
        self.steps = 0
        self.jitters = []

    def add_step(self, env: EnvT) -> None:
        self.steps += 1
        self.jitters.append(float(np.mean(abs(env.action - env.prev_action))))

    def summary(self) -> dict:
        result = {"steps": self.steps}
        result.update(self._describe("jitter", self.jitters))
        return result

    def _describe(self, name: str, values: list[float]) -> dict:
        return {
            f"{name}_mean": float(np.mean(values)),
            f"{name}_min": float(np.min(values)),
            f"{name}_max": float(np.max(values)),
        }
