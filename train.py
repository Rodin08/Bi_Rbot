from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from bi_rbot.tasks.locomotion.locomotion_env import LocomotionEnv

N_ENVS = 8
TOTAL_STEPS = 3000000
SAVE_EVERY = 100000
RUN_NAME = "PPO_" + datetime.now().strftime("%d,%m,%Y,%H:%M:%S")


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
        batch_size=256,
        verbose=1,
        tensorboard_log="runs/",
        device="cpu",
    )

    checkpoint = CheckpointCallback(
        SAVE_EVERY // N_ENVS, "models/" + RUN_NAME, "ppo", save_vecnormalize=True
    )
    model.learn(TOTAL_STEPS, checkpoint, tb_log_name=RUN_NAME, progress_bar=True)
    model.save("models/" + RUN_NAME + "/final")
    env.save("models/" + RUN_NAME + "/vecnormalize.pkl")


if __name__ == "__main__":
    main()
