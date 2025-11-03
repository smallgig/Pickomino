"""Test Pickomino."""

import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv

# from stable_baselines3 import SAC # Soft Actor-Critic (SAC) is suitable for continuous action spaces.

import pickomino_env  # Important: activates the registration.
from pickomino_env.src.bot import Bot

log_dir = "./logs/"
run_time = 0

# Create environment to test.
env = gym.make("Pickomino-v0", number_of_bots=2)  # base environment


def test_reset():
    """Test reset function."""
    observation, info = env.reset()
    assert np.array_equal(observation["dice_collected"], np.array([0, 0, 0, 0, 0, 0]))
    assert sum(observation["dice_rolled"]) == 8
    assert np.all(observation["tiles_table"] == 1)  # Initially [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert np.all(observation["tile_players"] == 0)  # Initially [0, 0, 0]


def test_step():
    """Test step function"""
    observation, info = env.reset()
    die_face_to_collect = np.argmax(observation["dice_rolled"])
    action = (die_face_to_collect, 0)  # Collect and keep rolling.
    observation, reward, terminated, truncated, info = env.step(action)
    assert sum(observation["dice_collected"]) + sum(observation["dice_rolled"]) == 8
    assert reward == 0
    assert not terminated
    assert not truncated
    assert info["player_stack"] == [42]  # 42 is initial invalid value for testing.


def test_multiple_actions():
    """Test multiple actions.

    Is a bit like play automated without printouts.
    """
    obs, info = env.reset()
    bot = Bot()
    total_reward = 0

    for step in range(3000):
        selection, stop = bot.policy(obs["dice_rolled"], obs["dice_collected"], int(str(info["smallest_tile"])))
        obs, reward, terminated, truncated, info = env.step((selection, stop))
        total_reward += reward
        # score = cast(Dice, game_info["dice"]).score()[0]
        if terminated:
            break
    print("\nTotal Reward:", total_reward)


def test_stable_baselines3():
    """Test environment works correctly with Stable Baselines3."""
    check_env(env)


def test_ppo():
    """Test PPO from stable_baselines3."""
    start_time = time.time()

    def make_env():
        """Create environment with function for testing PPO."""
        ppo_env = gym.make("Pickomino-v0", number_of_bots=2)
        ppo_env = Monitor(ppo_env, log_dir)  # Wrap with Stable Baselines 3's Monitor.
        return ppo_env

    # Vectorize environment for PPO for parallel environments
    vec_ppo_env = DummyVecEnv([make_env])

    # ent_coef for diversity of the policy through entropy
    # target_kl limits how far the policy may stray from the old status per update (trust-region).
    algorithm = PPO("MultiInputPolicy", vec_ppo_env)  # Add ', verbose=1' as necessary
    algorithm.learn(total_timesteps=100)  # 1 step = on action! (not episode!)

    end_time = time.time()
    global run_time
    run_time = end_time - start_time


def test_ppo_plotting():
    """Test PPO plotting."""
    x, y = ts2xy(load_results(log_dir), "timesteps")
    plt.plot(x, y)
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    plt.title(f"Learning Curve in {run_time:.0f} seconds")
    plt.grid()
    plt.show()


# def test_ppo_parallel():
#    """Test PPO parallel."""
#    start_time = time.time()
# from stable_baselines3.common.vec_env import SubprocVecEnv

# def make_env():
#     return gym.make("CartPole-v1")

# env = SubprocVecEnv([make_env for _ in range(8)])  # 8 parallel envs


if __name__ == "__main__":
    print("This file should be called with pytest.")
