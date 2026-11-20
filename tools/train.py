from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from bi_rbot.tasks.locomotion.locomotion_env import LocomotionEnv

N_ENVS = 8
TOTAL_STEPS = 5000000
SAVE_EVERY = 100000
RUN_NAME = "PPO_" + datetime.now().strftime("%d,%m,%Y,%H:%M:%S")

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RUNS = ROOT / "runs"


def main():

    env = VecNormalize(
        make_vec_env(
            LocomotionEnv,
            N_ENVS,
            vec_env_cls=SubprocVecEnv,
        )
    )

    model = PPO(
        "MlpPolicy",
        env,
        batch_size=4096,
        n_epochs=5,
        # policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        tensorboard_log=str(RUNS),
        device="cpu",
    )

    checkpoint = CheckpointCallback(
        SAVE_EVERY // N_ENVS, str(MODELS / RUN_NAME), "ppo", save_vecnormalize=True
    )
    model.learn(TOTAL_STEPS, checkpoint, tb_log_name=RUN_NAME, progress_bar=True)
    model.save(MODELS / RUN_NAME / "final")
    env.save(str(MODELS / RUN_NAME / "vecnormalize.pkl"))


if __name__ == "__main__":
    main()
