import numpy as np
from numpy.typing import NDArray

from bi_rbot.base.robot_env import RobotEnv
from bi_rbot.tasks.locomotion import commands
from bi_rbot.tasks.locomotion.locomotion_config import LocomotionConfig


class LocomotionEnv(RobotEnv):
    cfg: LocomotionConfig

    def __init__(self, cfg: LocomotionConfig | None = None, render_mode=None):
        self.cmd = np.zeros(3)
        super().__init__(
            cfg if cfg is not None else LocomotionConfig(), render_mode=render_mode
        )

    def _on_reset(self):
        self.cmd = commands.sample(self.np_random, self.cfg)

    def _on_step(self):
        if self.steps % self.cfg.resample_every == 0:
            self.cmd = commands.sample(self.np_random, self.cfg)

    def _get_obs(self) -> NDArray:
        return np.concatenate([super()._get_obs(), self.cmd])
