import gymnasium as gym
import mujoco
import numpy as np
from numpy.typing import NDArray

from bi_rbot.base import observations as observ
from bi_rbot.base.robot_config import RobotConfig


class RobotEnv(gym.Env):
    def __init__(self, cfg: RobotConfig | None = None, render_mode=None) -> None:
        self.cfg = cfg if cfg is not None else RobotConfig()
        self.model = mujoco.MjModel.from_xml_path(str(self.cfg.xml_path))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        self.body_id = self.model.body("Body").id
        self.leg_qpos = [
            self.model.joint(n).qposadr[0]
            for n in ("hip_R", "knee_R", "hip_L", "knee_L")
        ]
        self.leg_qvel = [
            self.model.joint(n).dofadr[0]
            for n in ("hip_R", "knee_R", "hip_L", "knee_L")
        ]
        self.wheel_qvel = [
            self.model.joint(n).dofadr[0] for n in ("wheel_R", "wheel_L")
        ]
        self.wheel_bodies = {self.model.body(n).id for n in ("RightWheel", "LeftWheel")}
        self.action = np.zeros(6)
        self.prev_action = np.zeros(6)
        self.filtered_action = np.zeros(6)
        self.prev_filtered_action = np.zeros(6)

        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=self._get_obs().shape, dtype=np.float64
        )

        self.action_space = gym.spaces.Box(-1, 1, shape=(6,), dtype=np.float32)

        self.steps = 0

    def _on_reset(self):
        pass

    def _on_step(self):
        pass

    def _get_obs(self) -> NDArray:
        return np.concatenate(
            [
                observ.get_body_up_axis(self),
                observ.get_gyro(self),
                observ.get_accel(self),
                observ.get_leg_joint_pos(self),
                observ.get_leg_joint_vel(self),
                observ.get_wheel_vel(self),
                self.action,
                self.filtered_action,
            ]
        )

    def _bad_ground_contact(self) -> bool:
        for contact in self.data.contact[: self.data.ncon]:
            participant_1 = self.model.geom_bodyid[contact.geom1]
            participant_2 = self.model.geom_bodyid[contact.geom2]
            if participant_1 == 0:
                other = participant_2
            elif participant_2 == 0:
                other = participant_1
            else:
                continue
            if other not in self.wheel_bodies:
                return True
        return False

    def _is_fallen(self) -> bool:
        return bool(
            observ.get_body_up_axis(self)[2] < self.cfg.min_straight_z
            or observ.get_body_height(self) < self.cfg.min_height
            or (self._bad_ground_contact() and self.cfg.terminate_on_body_contact)
        )

    def _compute_reward(self):
        total = 0.0
        terms = {}
        for name, (fn, weight) in self.cfg.reward_terms.items():
            value = fn(self)
            terms[name] = value
            total += weight * value
        return total, terms

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.action = np.zeros(6)
        self.prev_action = np.zeros(6)
        self.filtered_action = np.zeros(6)
        self._on_reset()
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        self.prev_action = self.action
        self.prev_filtered_action = self.filtered_action
        self.action = np.clip(action, -1, 1)
        self.filtered_action = (
            self.cfg.action_smoothing * self.action
            + (1 - self.cfg.action_smoothing) * self.filtered_action
        )
        ctrl = self.filtered_action * self.cfg.action_scale
        fwd, turn = self.filtered_action[4], self.filtered_action[5]
        ctrl[4] = np.clip(fwd + turn, -1, 1)  # wheel_R
        ctrl[5] = np.clip(fwd - turn, -1, 1)  # wheel_L
        self.data.ctrl[:] = ctrl
        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._on_step()
        obs = self._get_obs()
        reward, terms = self._compute_reward()
        terminated = self._is_fallen()
        truncated = self.steps >= self.cfg.max_steps

        return obs, reward, terminated, truncated, terms
