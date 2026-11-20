import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bi_rbot.tasks.locomotion.locomotion_env import LocomotionEnv
from bi_rbot.tasks.locomotion.locomotion_stats import LocomotionStats

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
EPISODES = 20


def main() -> None:
    if not MODELS.exists():
        sys.exit("No models/ folder, train first.")

    finished = [p for p in MODELS.iterdir() if (p / "final.zip").exists()]

    if not finished:
        sys.exit("No final.zip found")

    run = max(finished, key=lambda p: p.stat().st_mtime)

    raw_env = LocomotionEnv()
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(str(run / "vecnormalize.pkl"), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    obs = vec_env.reset()

    model = PPO.load(run / "final.zip")

    results = []
    for _ in range(EPISODES):
        stats = LocomotionStats()
        while True:
            assert isinstance(obs, np.ndarray)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = vec_env.step(action)
            if done[0]:
                break
            stats.add_step(raw_env)
        result = stats.summary()
        result["fell"] = not info[0].get("TimeLimit.truncated", False)
        results.append(result)

    print("Evaluating:", run.name)

    stats_keys = [k for k in results[0] if k.endswith(("_mean", "_min", "_max"))]
    names = dict.fromkeys(k.rsplit("_", 1)[0] for k in stats_keys)

    print(f"{'':20} {'mean':>8} {'min':>8} {'max':>8}")
    for name in names:
        row = [
            np.mean([r[f"{name}_{s}"] for r in results]) for s in ("mean", "min", "max")
        ]
        print(f"{name:20} {row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f}")

    print()
    for key in results[0]:
        if key not in stats_keys:
            print(f"{key:20} {np.mean([r[key] for r in results]):8.3f}")


if __name__ == "__main__":
    main()
