"""Pickomino game with gymnasium API."""

import random as rand
from typing import Optional
import numpy as np
import gymnasium as gym


class PickominoEnv(gym.Env):
    """The environment class."""

    def __init__(self, num_players):
        """Constructor."""
        self.num_dice = 8
        self.remaining_dice = self.num_dice
        self.num_players = num_players
        self.terminated = False
        self.truncated = False
        # Define what the agent can observe.
        # Dict space gives us structured, human-readable observations.
        # 6 possible faces of the dice. Worm = index 0, Rest: index = faces value of die
        self._dice_collected = np.array([0, 0, 0, 0, 0, 0])
        self._dice_rolled = np.array([0, 0, 0, 0, 0, 0])
        self.roll_counter = 0
        self.tile_table = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]

        self.observation_space = gym.spaces.Dict(
            {
                "dice_collected": gym.spaces.Discrete(n=6),
                "dice_rolled": gym.spaces.Discrete(n=6),
                "player": gym.spaces.Discrete(num_players),  # Number of players in the game.
                # "tiles": gym.spaces.Discrete(n=16, start=21) # Tiles that can be taken.
            }
        )
        # Action space is a tuple. First action: which dice you take. Second action: roll again or not.
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(6), gym.spaces.Discrete(2)))

    def _get_sum(self):
        """Return the sum of the collected dices."""
        return_value = 0

        # Dice with eyes = number of eyes per die times the number of dice.
        for ind, die in enumerate(self._dice_collected):
            return_value += ind * die  # Worms have ind = 0 and hence are not counted here.

        # Worms have value five.
        return_value += self._dice_collected[0] * 5

        return return_value

    def _get_obs_dice(self):
        """Convert internal state to observation format.

        Returns:
            tuple: Dices collected and dices rolled.
        """
        return self._dice_collected, self._dice_rolled

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
            "self._dice_collected": self._dice_collected,
            "self._dice_rolled": self._dice_rolled,
            "self.roll_counter": self.roll_counter,
            "self.observation_space": self.observation_space,
            "self.action_space": self.action_space,
            "self._get_sum()": self._get_sum(),
            "self._get_obs_dice()": self._get_obs_dice(),
            "self._get_obs_tiles()": self._get_obs_tiles(),
            "self.legal_move(action)": self.legal_move(action),
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

        self._dice_collected = np.array([0, 0, 0, 0, 0, 0])
        # self._dice_rolled = np.array([0, 2, 1, 4, 1, 0])
        # TODO: PyCharm suggested resetting dices_rolled and remaining_dice here as well

        for i in range(self.num_dice):
            self._dice_rolled[rand.randint(0, 5)] += 1

        self.remaining_dice = self.num_dice
        observation = self._get_obs_dice()
        info = self._get_info((0, 1))  # Arbitrary action in reset only for debugging

        return observation, info

    def legal_move(self, action):
        """Check if action is allowed."""
        terminated = False
        truncated = False

        if action[1] == 0:
            truncated = True
        # TODO: This is wrong! There is no limit to the number of rolls of the dice.
        # Terminated should be when a misthrow occurs

        # Dice already collected cannot be taken again.
        elif self.roll_counter >= 2:
            terminated = True
            for die in self._dice_rolled:
                if die not in self._dice_collected:
                    terminated = False

        if self._dice_collected[0] == 0 and self.remaining_dice == 0:
            terminated = True

        return terminated, truncated

    def step_dice(self, action):
        """Execute one roll of the dice and picking or returning a tile.

        Args:
            action: The action to take: which dice to collect.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # TODO: currently only the roll of the dice is implemented. The tile moving phase is not implemented yet.
        self.terminated, self.truncated = self.legal_move(action)

        if self.terminated or self.truncated:
            self._dice_collected = np.array([0, 0, 0, 0, 0, 0])
            self._dice_rolled = np.array([0, 0, 0, 0, 0, 0])
            self.roll_counter = 0
            self.remaining_dice = 8
            # TODO: PyCharm suggests resetting remaining_dice here.
            # TODO: Check: if terminated or truncated should we not stop updating dice values completely??
        else:
            self._dice_collected[action[0]] = self._dice_rolled[action[0]]
            # Reduce the remaining number of dice by the number collected.
            self.remaining_dice -= self._dice_collected[action[0]]
            self._dice_rolled = np.array([0, 0, 0, 0, 0, 0])

            if action[1]:
                max_dice = self.num_dice - np.sum(self._dice_collected)
                dices_to_roll = min(self.remaining_dice, max_dice)

                for i in range(dices_to_roll):
                    self._dice_rolled[rand.randint(0, 5)] += 1

            self.roll_counter += 1

    def step_tiles(self):
        dice_sum = self._get_sum()
        tile_test = []
        # Environment takes the highest tile on the table or from a player.
        if dice_sum >= 21:
            self.tile_table.remove(dice_sum)
            tile_test.append(dice_sum)
            print("Your tiles:", tile_test)
            self.truncated = True
        else:
            self.truncated = False

    def step(self, action):

        self.step_dice(action)
        self.step_tiles()

        observation = self._get_obs_dice()
        info = self._get_info(action)
        reward = self._get_sum()

        return observation, reward, self.terminated, self.truncated, info


if __name__ == "__main__":
    env = PickominoEnv(2)
    observation, info = env.reset()
    print("Reset", observation)
    for key, value in info.items():
        print(key, value)
    print("--------------------")

    taken = []
    print(observation)
    for step in range(6):

        selection = int(np.argmax(observation[1]))

        # Do not select the same die face value again
        if selection in taken:
            observation[1][selection] = 0
            selection = int(np.argmax(observation[1]))

        taken.append(selection)
        print("step:", step, "    selection:", selection, "   taken:", taken)
        action = (selection, 1)
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"act: {action}, obs(col, rol): {observation}, rew: {reward}, ter: {terminated}, tru: {truncated}")
        for key, value in info.items():
            print(key, value)
        print("--------------------")
