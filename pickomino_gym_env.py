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
        self.player_tiles = []
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

    def _soft_reset(self):
        self._dice_collected = np.array([0, 0, 0, 0, 0, 0])
        self._dice_rolled = np.array([0, 0, 0, 0, 0, 0])
        self.roll_counter = 0
        self.remaining_dice = 8

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
        self._dice_rolled = np.array([0, 0, 0, 0, 0, 0])
        self.roll_counter = 0
        self.remaining_dice = 8
        self.player_tiles = []
        self.tile_table = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
        self.terminated = False
        self.truncated = False

        # self._dice_rolled = np.array([0, 2, 1, 4, 1, 0])

        for _ in range(self.num_dice):
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
        # Terminated should be when a misthrow occurs

        # Dice already collected cannot be taken again.
        elif self.roll_counter >= 2:
            self.terminated = True
            # TODO: Jarl: still think this is not right. It compares all combinations = 36 checks. But only 6 checks
            # are needed. Consider:
            # for index in range(len(self._dice_rolled)):
            # #### if dice_rolled[index] > 0 and die_collected[index] == 0:
            # #### ##### self.terminated = False
            for die_collected in self._dice_collected:
                for die_rolled in self._dice_rolled:
                    if die_rolled > 0 and die_collected == 0:
                        self.terminated = False
        # TODO: Jarl: think it should be ... and self._get_sum() < 21  <- strictly less than 21 NOT less than or equal
        # as the 'smallest' tile is 21 and hence can be taken if the sum is equal (==) to 21
        if self.remaining_dice == 0 and self._get_sum() <= 21:
            self.terminated = True

        if self.remaining_dice == 0 and self._dice_collected[0] == 0:
            self.terminated = True

            # TODO: Jarl: can this commented out code be deleted now?
            # terminated = True
            # for die in self._dice_rolled:
            #     if die not in self._dice_collected:
            #         terminated = False
            #         break

        return terminated, truncated

    def step_dice(self, action):
        """Execute one roll of the dice and picking or returning a tile.

        :param: action: The action to take: which dice to collect.
        """
        self.terminated, self.truncated = self.legal_move(action)

        if self.terminated or self.truncated:
            self._soft_reset()
            # TODO: Check: if terminated or truncated should we not stop updating dice values completely??
        else:
            self._dice_collected[action[0]] = self._dice_rolled[action[0]]
            # Reduce the remaining number of dice by the number collected.
            self.remaining_dice -= self._dice_collected[action[0]]
            self._dice_rolled = np.array([0, 0, 0, 0, 0, 0])

            if action[1]:
                max_dice = self.num_dice - np.sum(self._dice_collected)
                dices_to_roll = min(self.remaining_dice, max_dice)

                for _ in range(dices_to_roll):
                    self._dice_rolled[rand.randint(0, 5)] += 1

            self.roll_counter += 1

    def step_tiles(self):
        """Internal step for picking or returning a tile."""
        dice_sum = self._get_sum()
        # Environment takes the highest tile on the table or from a player.
        if dice_sum >= 21:
            self.tile_table.remove(dice_sum)
            self.player_tiles.append(dice_sum)
            print("Your tiles:", self.player_tiles)
            self.truncated = True
            self._soft_reset()
        else:
            self.truncated = False

    def step(self, action):

        self.step_dice(action)
        self.step_tiles()

        observation = self._get_obs_dice()
        info = self._get_info(action)
        reward = self._get_sum()

        return observation, reward, self.terminated, self.truncated, info


# The next 18 lines, until 'print(*line)', were copied from Stack Overflow


die_faces = [
    "",  # index = 0 doesn't have a face
    "[     ]\n[  0  ]\n[     ]",  # index 1
    "[0    ]\n[     ]\n[    0]",  # index 2
    "[0    ]\n[  0  ]\n[    0]",  # index 3
    "[0   0]\n[     ]\n[0   0]",  # index 4
    "[0   0]\n[  0  ]\n[0   0]",  # index 5
    "[0   0]\n[0   0]\n[0   0]",  # index 6
]


def print_dice(values: list[int]):
    """Print one dice."""
    faces = [die_faces[v].splitlines() for v in values]
    for line in zip(*faces):
        print(*line)


def print_roll(observation, reward):
    """Print one roll."""
    print_dice([1, 2, 3, 4, 5, 6])
    for i in range(len(observation[0]) - 1):
        print(f"   {observation[0][i+1]}    ", end="")
    print(f"   {observation[0][0]}    ", end="")
    print(f" collected   sum = {reward}")
    for i in range(len(observation[1]) - 1):
        print(f"   {observation[1][i+1]}    ", end="")
    print(f"   {observation[1][0]}    ", end="")
    print(" rolled")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    NUMBER_OF_DICE: int = 8
    NUMBER_OF_PLAYERS = 2
    env = PickominoEnv(NUMBER_OF_PLAYERS)
    """Interactive test."""
    observation, info = env.reset()
    reward = 0
    print("Reset")
    print()
    for step in range(6):
        print_roll(observation, reward)
        # action = (0, 1)  # dummy
        # print(f"act: {action}")
        # for key, value in info.items():
        #    print(key, value)
        # print("--------------------")
        selection = int(input("Which dice do you want to collect? (1..5 or worm =6) or -1 to stop: "))
        # print("step:", step, "    selection:", selection)
        if selection == -1:
            break
        if selection == 6:
            selection = 0
        action = (selection, 1)
        observation, reward, terminated, truncated, info = env.step(action)
