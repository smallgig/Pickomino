"""Test Pickomino."""

import numpy as np
from pickomino_env.pickomino_gym_env import PickominoEnv

# Create environment to test.
env = PickominoEnv(2)


def test__init__():
    """Test init function."""
    assert env._number_of_bots == 2


def test_reset():
    """Test reset function."""
    observation, info = env.reset()
    assert np.array_equal(observation["dice_collected"], np.array([0, 0, 0, 0, 0, 0]))
    assert sum(observation["dice_rolled"]) == 8


def test_step():
    """Test step function"""
    observation, info = env.reset()
    die_face_to_collect = np.argmax(observation["dice_rolled"])
    action = (die_face_to_collect, 0)  # Collect and keep rolling.
    observation, reward, terminated, truncated, info = env.step(action)
    assert sum(observation["dice_collected"]) + sum(observation["dice_rolled"]) == 8


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


def run_all_test():
    """Run all tests."""
    test__init__()
    test_reset()
    test_step()
    test_multiple_actions()


if __name__ == "__main__":
    env = PickominoEnv(2)
    run_all_test()
