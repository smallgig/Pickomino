"""Pickomino game with gymnasium API."""
from typing import Optional
import numpy as np
import gymnasium as gym
import random as rand


class PickominoEnv(gym.Env):
    """The environment class."""

    def __init__(self, num_players):
        """Constructor."""
        self.num_dice = 8
        self.remaining_dice = self.num_dice
        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self._dices_collected = np.array([0, 0, 0, 0, 0, 0])
        self._dices_rolled = np.array([0, 0, 0, 0, 0, 0])
        self.roll_counter = 0
        self.observation_space = gym.spaces.Dict(
            {
                "dices_collected": gym.spaces.Discrete(n=6),
                "dices_rolled": gym.spaces.Discrete(n=6),
                "player": gym.spaces.Discrete(num_players),  # Number of players in the game.
                # "tiles": gym.spaces.Discrete(n=16, start=21) # Tiles that can be taken.
            }
        )
        # Action space is a tuple. First action: which dice you take. Second action: roll again or not.
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(6), gym.spaces.Discrete(2)))

    def _get_sum(self, collected_dices):
        """Return the sum of the collected dices."""
        return_value = 0

        # Dice with eyes = number of eyes per dice times the number of dices.
        for ind, dice in enumerate(collected_dices):
            return_value += dice * ind

        # Worms have value five.
        return_value = return_value + collected_dices[0] * 5

        return return_value

    def _get_obs_dices(self):
        """Convert internal state to observation format.

        Returns:
            tuple: Dices collected and dices rolled.
        """
        return self._dices_collected, self._dices_rolled

    def _get_obs_tiles(self):
        """Convert internal state to observation format.

        Returns:
            dict: Tiles distribution
        """
        # TODO: implement correct return value
        return self.observation_space

    def _get_info(self, action):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with additional information which is useful for debugging but not necessary for learning.
        """
        return_value = {
            "action": action,
            "self.num_dice": self.num_dice,
            "self.remaining_dice": self.remaining_dice,
            "self._dices_collected": self._dices_collected,
            "self._dices_rolled": self._dices_rolled,
            "self.roll_counter": self.roll_counter,
            "self.observation_space": self.observation_space,
            "self.action_space": self.action_space,
            "self._get_sum(self._dices_collected)": self._get_sum(self._dices_collected),
            "self._get_obs_dices()": self._get_obs_dices(),
            "self._get_obs_tiles()": self._get_obs_tiles(),
            "self.legal_move(action)": self.legal_move(action)
        }
        return return_value

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

        self._dices_collected = np.array([0, 0, 0, 0, 0, 0])

        for i in range(self.num_dice):
            self._dices_rolled[rand.randint(0, 5)] += 1

        self.remaining_dice = self.num_dice
        observation = self._get_obs_dices()
        info = self._get_info((0, 1))  # Arbitrary action in reset only for debugging

        return observation, info

    def legal_move(self, action):
        """Check if action is allowed."""
        terminated = False
        truncated = False

        if action[1] == 0:
            truncated = True
        elif self.roll_counter == 3:
            terminated = True

        return terminated, truncated

    def step(self, action):
        """Execute one roll of the dices and picking or returning a tile.

        Args:
            action: The action to take: which dice to collect.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # TODO: currently only the roll of the dices is implemented. The tile moving phase is not implemented yet.
        terminated, truncated = self.legal_move(action)

        if terminated or truncated:
            self._dices_collected = np.array([0, 0, 0, 0, 0, 0])
            self._dices_rolled = np.array([0, 0, 0, 0, 0, 0])
            self.roll_counter = 0

        self._dices_collected[action[0]] = self._dices_rolled[action[0]]
        # Reduce the remaining number of dices by the number collected.
        self.remaining_dice -= self._dices_collected[action[0]]
        self._dices_rolled = np.array([0, 0, 0, 0, 0, 0])

        if action[1]:
            for i in range(self.remaining_dice):
                self._dices_rolled[rand.randint(0, 5)] += 1

        reward = self._get_sum(self._dices_collected)

        observation = self._get_obs_dices()

        self.roll_counter += 1
        info = self._get_info(action)
        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    env = PickominoEnv(2)
    observation, info = env.reset()
    print("Reset", observation)
    for key, value in info.items():
        print(key, value)
    print("--------------------")

    for step in range(6):
        selection = int(np.argmax(observation[1]))
        print("step", step, "    selection", selection)
        action = (selection, 1)
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"act: {action}, obs: {observation}, rew: {reward}, ter: {terminated}, tru: {truncated}")
        # for key, value in info.items():
        #     print(key, value)
        print("--------------------")
