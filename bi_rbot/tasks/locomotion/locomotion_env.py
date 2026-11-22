import numpy as np
from numpy.typing import NDArray

from bi_rbot.base.robot_env import RobotEnv
from bi_rbot.tasks.locomotion import commands
from bi_rbot.tasks.locomotion.locomotion_config import LocomotionConfig


class LocomotionEnv(RobotEnv):
    cfg: LocomotionConfig

    def __init__(self, cfg: LocomotionConfig | None = None, render_mode=None):
        # must exist before super().__init__(), which calls _get_obs() to size the obs space
        self.cmd = np.zeros(len(commands.Commands))

        super().__init__(
            cfg if cfg is not None else LocomotionConfig(), render_mode=render_mode
        )

        self.fixed_cmd: NDArray | None = None

    # --- Step pipeline ---
    def _get_obs(self) -> NDArray:
        return np.concatenate([super()._get_obs(), self.cmd])

    # --- Hooks for tasks ---
    def _on_reset(self):
        self.cmd = self._next_cmd()

    def _on_step(self):
        if self.fixed_cmd is not None or self.steps % self.cfg.resample_every == 0:
            self.cmd = self._next_cmd()

    def _next_cmd(self) -> NDArray:
        if self.fixed_cmd is not None:
            return self.fixed_cmd.copy()
        return commands.sample(self.np_random, self.cfg)
