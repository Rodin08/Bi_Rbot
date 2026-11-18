import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bi_rbot.tasks.locomotion.locomotion_env import LocomotionEnv

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def main():

    if not MODELS.exists():
        sys.exit("No models/ folder, train first.")

    finished = [p for p in MODELS.iterdir() if (p / "final.zip").exists()]

    if not finished:
        sys.exit("No final.zip found")

    run = max(finished, key=lambda p: p.stat().st_mtime)
    print("Watching:", run)

    raw_env = LocomotionEnv()
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(str(run / "vecnormalize.pkl"), vec_env)
    terms = raw_env.cfg.reward_terms
    sums = {name: 0.0 for name in terms}
    steps = 0
    vec_env.training = False
    vec_env.norm_reward = False
    obs = vec_env.reset()

    model = PPO.load(run / "final.zip")

    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
        while viewer.is_running():
            assert isinstance(obs, np.ndarray)
            action, _ = model.predict(observation=obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            steps += 1
            for name, (_, weight) in terms.items():
                sums[name] += weight * info[0][name]

            if done[0]:
                print(f"\nEpisode: {steps} steps")
                for name, (_, weight) in terms.items():
                    best = f"of max {weight * steps:.0f}" if weight > 0 else "(penalty)"
                    print(f"  {name:16} {sums[name]:8.1f}  {best}")
                sums = {name: 0.0 for name in terms}
                steps = 0
            viewer.sync()
            time.sleep(raw_env.cfg.frame_skip * raw_env.model.opt.timestep)


if __name__ == "__main__":
    main()
