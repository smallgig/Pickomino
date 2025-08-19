from torch.ao.quantization.backend_config.onednn import observation_type

import pickomino_gym_env
import numpy as np


def test_basics():
    action = (0, 1)
    observation, reward, terminated, truncated, info = env.step(action)

    assert True


def test_reset():
    observation, info = env.reset()
    print(observation)

    assert np.array_equal(observation[0], np.array([0, 0, 0, 0, 0, 0]))
    assert np.array_equal(observation[1], np.array([0, 2, 1, 4, 1, 0]))


def test_multiple_actions():
    observation, info = env.reset()

    taken = []
    for step in range(6):
        selection = int(np.argmax(observation[1]))

        # Do not select the same die face value again
        if selection in taken:
            observation[1][selection] = 0
            selection = int(np.argmax(observation[1]))
            observation, reward, terminated, truncated, info = env.step(selection)

    assert


env = pickomino_gym_env.PickominoEnv(2)
test_basics()
