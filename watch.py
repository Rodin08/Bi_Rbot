import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bi_rbot.base.robot_env import RobotEnv


def main():
    models = Path("models")

    if not models.exists():
        sys.exit(
            "No models/ folder. Start watch.py from inside Bi_rBot/, or train first."
        )

    finished = [p for p in models.iterdir() if (p / "final.zip").exists()]

    if not finished:
        sys.exit("No final.zip found")

    run = max(finished, key=lambda p: p.stat().st_mtime)
    print("Watching:", run)

    raw_env = RobotEnv()
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(str(run / "vecnormalize.pkl"), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    obs = vec_env.reset()

    model = PPO.load(run / "final.zip")

    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
        while viewer.is_running():
            assert isinstance(obs, np.ndarray)
            action, _ = model.predict(observation=obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            viewer.sync()
            time.sleep(raw_env.cfg.frame_skip * raw_env.model.opt.timestep)


if __name__ == "__main__":
    main()
