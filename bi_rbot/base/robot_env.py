import math

import gymnasium as gym
import mujoco
import numpy as np
from numpy.typing import NDArray

from bi_rbot.base import kinematics
from bi_rbot.base import observations as observ
from bi_rbot.base.actions import Actions, decode_actions
from bi_rbot.base.robot_config import RobotConfig

LEG_JOINTS = ("hip_r", "knee_r", "hip_l", "knee_l")
WHEEL_JOINTS = ("wheel_r", "wheel_l")
WHEEL_BODIES = ("right_wheel", "left_wheel")
ACTUATORS = ("hip_r", "hip_l", "knee_r", "knee_l", "wheel_r", "wheel_l")
WORLD_BODY = 0  # MuJoCo always gives the world (floor) body id 0


class RobotEnv(gym.Env):
    def __init__(self, cfg: RobotConfig | None = None, render_mode=None) -> None:
        # --- Config and simulation ---
        self.cfg = cfg if cfg is not None else RobotConfig()
        self.model = mujoco.MjModel.from_xml_path(str(self.cfg.xml_path))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        # --- Model lookups (names → indices, once) ---
        self.body_id = self.model.body("body").id
        self.leg_qpos = [self.model.joint(n).qposadr[0] for n in LEG_JOINTS]
        self.leg_qvel = [self.model.joint(n).dofadr[0] for n in LEG_JOINTS]
        self.wheel_qvel = [self.model.joint(n).dofadr[0] for n in WHEEL_JOINTS]
        self.wheel_bodies = {self.model.body(n).id for n in WHEEL_BODIES}
        self.actuator_ids = {n: self.model.actuator(n).id for n in ACTUATORS}

        # --- Episode state ---
        self._reset_state()

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=self._get_obs().shape, dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            -1, 1, shape=(len(Actions),), dtype=np.float32
        )

    # --- Gym interface ---
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._reset_state()
        self._on_reset()
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        self._apply_action(action)
        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._on_step()
        obs = self._get_obs()
        reward, terms = self._compute_reward()
        terminated = self._is_fallen()
        truncated = self.steps >= self.cfg.max_steps

        return obs, reward, terminated, truncated, terms

    # --- Step pipeline ---
    def _apply_action(self, action) -> None:
        self.prev_action = self.action
        self.action = np.clip(action, -1, 1)
        self.filtered_action = (
            self.cfg.action_smoothing * self.action
            + (1 - self.cfg.action_smoothing) * self.filtered_action
        )
        decoded = decode_actions(self.filtered_action, self.cfg)

        leg_r = kinematics.leg_ik(decoded.leg_r, self.cfg)
        leg_l = kinematics.leg_ik(decoded.leg_l, self.cfg)

        targets = {
            "hip_r": leg_r.hip,
            "hip_l": leg_l.hip,
            "knee_r": leg_r.knee,
            "knee_l": leg_l.knee,
            "wheel_r": decoded.wheel_r,
            "wheel_l": decoded.wheel_l,
        }
        for name, value in targets.items():
            self.data.ctrl[self.actuator_ids[name]] = value

    def _get_obs(self) -> NDArray:
        return np.concatenate(
            [
                observ.get_projected_up(self),
                observ.get_gyro(self),
                observ.get_wheel_vel(self),
                observ.get_leg_length(self),
                observ.get_leg_angle(self),
                observ.get_leg_length_vel(self),
                observ.get_leg_angle_vel(self),
                self.filtered_action,
            ]
        )

    def _compute_reward(self):
        total = 0.0
        terms = {}
        for name, (fn, weight) in self.cfg.reward_terms.items():
            value = fn(self)
            terms[name] = value
            total += weight * value
        return total, terms

    def _is_fallen(self) -> bool:
        return bool(
            observ.get_projected_up(self)[2] < math.cos(math.radians(self.cfg.max_tilt_deg))
            or observ.get_body_height(self) < self.cfg.min_height
            or (self._bad_ground_contact() and self.cfg.terminate_on_body_contact)
        )

    def _bad_ground_contact(self) -> bool:
        for contact in self.data.contact[: self.data.ncon]:
            participant_1 = self.model.geom_bodyid[contact.geom1]
            participant_2 = self.model.geom_bodyid[contact.geom2]
            if participant_1 == WORLD_BODY:
                other = participant_2
            elif participant_2 == WORLD_BODY:
                other = participant_1
            else:
                continue
            if other not in self.wheel_bodies:
                return True
        return False

    # --- State ---
    def _reset_state(self) -> None:
        self.steps = 0
        self.action = np.zeros(len(Actions))
        self.prev_action = np.zeros(len(Actions))
        self.filtered_action = np.zeros(len(Actions))

    # --- Hooks for tasks ---
    def _on_reset(self):
        pass

    def _on_step(self):
        pass
