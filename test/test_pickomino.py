"""Test Pickomino."""

import os
import time
import numpy as np
import pytest
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv

# For later
# from stable_baselines3 import SAC # Soft Actor-Critic (SAC) is suitable for continuous action spaces.

import pickomino_env  # noqa:F401 # side-effect import, required for environment registration.
from pickomino_env.src.bot import Bot

# Create an environment to test.
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
    assert info["player_stack"] == [42]  # 42 is an initial invalid value for testing.


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
    # print("\nTotal Reward:", total_reward)


def test_stable_baselines3():
    """Test environment works correctly with Stable Baselines3."""
    check_env(env)


@pytest.fixture(scope="session")
def ppo_setup(tmp_path_factory):
    """Fixture for creating a PPO model from Stable Baselines3."""
    log_dir = tmp_path_factory.mktemp("ppo_logs")

    def make_env(rank):
        """Create environment with function for testing PPO."""

        def _init():
            ppo_env = gym.make("Pickomino-v0", number_of_bots=6)
            ppo_env = Monitor(ppo_env, filename=f"{log_dir}/{rank}")
            return ppo_env

        return _init

    # More advanced, hence for later.
    # from stable_baselines3.common.vec_env import SubprocVecEnv
    # par_env = SubprocVecEnv([make_env for _ in range(8)])  # 8 parallel envs.

    # Vectorize environment for PPO for parallel environments
    par_env = DummyVecEnv([make_env(i) for i in range(8)])  # 8 sequential envs for debugging.
    model = PPO("MultiInputPolicy", par_env)  # Add ', verbose=1' as necessary

    start_time = time.time()
    # 1 step = one action! (not an episode!).
    # model.learn(total_timesteps=500_000)  # Noticeable learning in 15 minutes on my machine.
    # model.learn(total_timesteps=10_000_000)  # Going for real learning. Use assert ppo_run_time < 8 * 60 * 60.
    model.learn(total_timesteps=10000)  # Run fast
    ppo_run_time = time.time() - start_time
    assert ppo_run_time < 12000, f"PPO training took too long: {ppo_run_time:.0f} seconds"
    return model, log_dir, ppo_run_time


def test_ppo_plotting(ppo_setup):
    """Test PPO result plotting (non-interactive)."""
    model, log_dir, ppo_run_time = ppo_setup
    print("Plotting... with time: ", ppo_run_time)
    x, y = ts2xy(load_results(str(log_dir)), "timesteps")

    # Simple validation: no crash and x, y are arrays
    assert x is not None
    assert y is not None

    # Plot (non-blocking)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    plt.title(f"Learning Curve in {ppo_run_time:.0f} seconds")
    plt.grid(True)
    os.makedirs("test/plots", exist_ok=True)
    plt.savefig("test/plots/PPO_learning_curve.png")
    plt.close()

    assert len(x) > 0, "No training data found in monitor logs."


def test_action_out_of_range():
    """Test action out of range."""
    env.reset()
    obs, reward, term, trunc, info = env.step(env.action_space.sample())  # Should be good.
    assert not term
    obs, reward, term, trunc, info = env.step((7, 0))  # Face out of range
    assert term
    env.reset()
    obs, reward, term, trunc, info = env.step(env.action_space.sample())  # Should be good.
    assert not term
    obs, reward, term, trunc, info = env.step((2, 4))  # Roll out of range
    assert term


if __name__ == "__main__":
    print("This file should be called with pytest.")
