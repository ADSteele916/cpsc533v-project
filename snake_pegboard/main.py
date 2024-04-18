import argparse

from snake_env import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


def set_model_with_gravity(model, g: float):
    vec_env = make_vec_env(
        "Snake",
        n_envs=8,
        seed=4,
        env_kwargs=dict(
            snake_config=SnakeConfig(12, gravity=g),
            render_mode="rgb_array",
            width=854,
            height=480,
        ),
        vec_env_cls=SubprocVecEnv,
    )
    model.set_env(vec_env)
    return vec_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gravities", nargs="+", type=float, help="Gravities at each generation")
    args = parser.parse_args()
    gravities = args.gravities
    gravity = gravities[0]

    vec_env = make_vec_env(
        "Snake",
        n_envs=8,
        seed=4,
        env_kwargs=dict(
            snake_config=SnakeConfig(12, gravity=gravity * -9.80665),
            render_mode="rgb_array",
            width=854,
            height=480,
        ),
        vec_env_cls=SubprocVecEnv,
    )
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", tensorboard_log="./tensorboard/")
    model.learn(total_timesteps=1000000, tb_log_name="curriculum-snake-pegboard", reset_num_timesteps=False)
    model.save(f"ppo_snake_01M_{gravities[0]}g.zip")
    for i, gravity in enumerate(gravities[1:]):
        set_model_with_gravity(model, -9.80665 * gravity)
        model.learn(total_timesteps=1000000, tb_log_name="curriculum-snake-pegboard", reset_num_timesteps=False)
        model.save(f"ppo_snake_{i:02}M_{gravity}g.zip")

    vec_env = make_vec_env(
        "Snake",
        n_envs=1,
        seed=4,
        env_kwargs=dict(
            snake_config=SnakeConfig(12, gravity=gravity),
            render_mode="human",
        ),
    )
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")


if __name__ == "__main__":
    main()
