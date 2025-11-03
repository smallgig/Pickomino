"""Test Pickomino."""

import gymnasium as gym
import numpy as np

import pickomino_env  # Important: activates the registration.

# Create environment to test.
env = gym.make("Pickomino-v0", number_of_bots=2)


# def test__init__():
#     """Test init function."""
#     assert env._number_of_bots == 2


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
    """Test multiple actions."""
    pass
    # observation, info = env.reset()
    #
    # taken = []
    # for step in range(6):
    #     selection = int(np.argmax(observation['dice_rolled']))
    #
    #     # Do not select the same die face value again
    #     if selection in taken:
    #         observation['dice_collected'][selection] = 0
    #         selection = int(np.argmax(observation['dice_rolled']))
    #         observation, reward, terminated, truncated, info = env.step((selection, 0))


if __name__ == "__main__":
    print("This file should be called with pytest.")
