from bi_rbot.base import observations as observ
from bi_rbot.base.robot_stats import EpisodeStats
from bi_rbot.tasks.locomotion.locomotion_env import LocomotionEnv


class LocomotionStats(EpisodeStats[LocomotionEnv]):
    def __init__(self) -> None:
        super().__init__()
        self.height_errors = []
        self.speed_errors = []
        self.yaw_errors = []

    def add_step(self, env: LocomotionEnv) -> None:
        super().add_step(env)
        self.height_errors.append(abs(observ.get_body_height(env) - env.cmd[0]))
        self.speed_errors.append(abs(observ.get_forward_vel(env) - env.cmd[1]))
        self.yaw_errors.append(abs(observ.get_yaw_rate(env) - env.cmd[2]))

    def summary(self) -> dict:
        result = super().summary()
        result.update(self._describe("height", self.height_errors))
        result.update(self._describe("speed", self.speed_errors))
        result.update(self._describe("yaw", self.yaw_errors))
        return result
