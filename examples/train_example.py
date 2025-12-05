"""Example: Train a Reinforcement Learning Agent on Pickomino environment."""

import gymnasium as gym
import pickomino_env  # noqa: F401 Triggers registration

env = gym.make("Pickomino-v0", render_mode=None, number_of_bots=2)
obs, info = env.reset(seed=42)

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated:
        obs, info = env.reset()

env.close()
