from snake_env.snake import SnakeEnv, SnakeConfig

import gymnasium as gym

gym.register(
    id="Snake",
    entry_point="snake_env.snake:SnakeEnv",
    max_episode_steps=10000,
    reward_threshold=30.0,
)
