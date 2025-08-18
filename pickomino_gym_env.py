from typing import Optional
import numpy as np
import gymnasium as gym
import random as rand

from markdown.extensions.smarty import remainingDoubleQuotesRegex
from torchrl.envs import terminated_or_truncated


class PickominoEnv(gym.Env):

    def __init__(self, num_players):
        self.num_dice = 8
        self.remaining_dice = self.num_dice
        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self._dices_collected = np.array([0, 0, 0, 0, 0, 0])
        self._dices_rolled = np.array([0, 0, 0, 0, 0 ,0])
        self.roll_counter = 0


        self.observation_space = gym.spaces.Dict(
            {
                "dices_collected": gym.spaces.Discrete(n=6, start=0),
                "dices_rolled": gym.spaces.Discrete(n=6,start=0),
                "player": gym.spaces.Discrete(num_players), # Players in game
                # "tiles": gym.spaces.Discrete(n=16, start=21) # Tiles that can be taken
            }
        )
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(6), gym.spaces.Discrete(2))) # First Action which dice u take,
                                                                             # Second Action Roll again or not

    def _get_sum(self, collected_dices):
        return_value = 0
        for ind, dice in enumerate(collected_dices):
            return_value += dice * ind

        return_value = return_value + collected_dices[0] * 5

        return return_value

    def _get_obs_dices(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return self._dices_collected, self._dices_rolled

    def _get_obs_tiles(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return self.observation_space

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self._dices_collected = np.array([0, 0, 0, 0, 0, 0])
        dice_list = [0, 1, 2, 3, 4, 5]
        for i in range(self.num_dice):
            self._dices_rolled[rand.randint(0, 5)] += 1

        self.remaining_dice = self.num_dice
        observation = self._get_obs_dices()
        info = self._get_info()

        return observation, info

    def legal_move(self, action):
        terminated = False
        truncated = False

        if action[1] == 0:
            truncated = True
        elif self.roll_counter == 3:
            terminated = True

        return terminated, truncated


    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        terminated, truncated = self.legal_move(action)

        if terminated or truncated:
            self._dices_collected = np.array([0, 0, 0, 0, 0, 0])
            self._dices_rolled = np.array([0, 0, 0, 0, 0, 0])
            self.roll_counter = 0


        self._dices_collected[action[0]] = self._dices_rolled[action[0]]
        self._dices_rolled = np.array([0, 0, 0, 0, 0 ,0])

        if action[1]:
            for i in range(self.remaining_dice):
                self._dices_rolled[rand.randint(0, 5)] += 1
                self.remaining_dice -= 1

        reward = self._get_sum(self._dices_collected)

        observation = self._get_obs_dices()
        info = self._get_info()

        self.roll_counter += 1

        return observation, reward, terminated, truncated, info

if __name__ == "__main__":
    env = PickominoEnv(2)
    observation, info = env.reset()
    print(observation)
    observation, reward, _, _, _ = env.step(action=(5, 1))
    print(observation, reward)
    observation, reward, _, _, _ = env.step(action=(3, 1))
    print(observation, reward)
    observation, reward, _, _, _ = env.step(action=(2, 1))
    print(observation, reward)
    observation, reward, _, _, _ = env.step(action=(1, 1))
    print(observation, reward)
    observation, reward, _, _, _ = env.step(action=(5, 1))
    print(observation, reward)
    observation, reward, _, _, _ = env.step(action=(5, 1))
    print(observation, reward)
    observation, reward, _, _, _ = env.step(action=(5, 1))
    print(observation, reward)
    observation, reward, _, _, _ = env.step(action=(5, 1))
    print(observation, reward)
    observation, reward, _, _, _ = env.step(action=(5, 1))
    print(observation, reward)

