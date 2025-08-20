"""Pickomino game with gymnasium API."""

import random as rand
from typing import Optional
import numpy as np
import gymnasium as gym


class PickominoEnv(gym.Env):
    """The environment class."""

    def __init__(self, num_players: int) -> None:
        """Constructor."""
        self.action_index_dice = 0
        self.action_index_roll = 1
        self.action_roll = 0
        self.action_stop = 1
        self.num_dice: int = 8
        self.remaining_dice: int = self.num_dice
        self.num_players: int = 2
        self.terminated: bool = False
        self.truncated: bool = False
        # Define what the agent can observe.
        # Dict space gives us structured, human-readable observations.
        # 6 possible faces of the dice. Worm = index 0, Rest: index = faces value of die
        self._dice_collected: list[int] = [0, 0, 0, 0, 0, 0]
        self._dice_rolled: list[int] = [0, 0, 0, 0, 0, 0]
        self.roll_counter: int = 0
        self.tile_table = {
            21: True,
            22: True,
            23: True,
            24: True,
            25: True,
            26: True,
            27: True,
            28: True,
            29: True,
            30: True,
            31: True,
            32: True,
            33: True,
            34: True,
            35: True,
            36: True,
        }
        self.you: dict[int, int] = {}

        self.observation_space = gym.spaces.Dict(
            {
                "dice_collected": gym.spaces.Box(low=0, high=6, shape=(6,), dtype=np.int64),
                "dice_rolled": gym.spaces.Box(low=0, high=6, shape=(6,), dtype=np.int64),
                "tiles_table": gym.spaces.Dict(
                    {i: gym.spaces.Discrete(5) for i in range(21, 37)}
                ),
                "tiles_player": gym.spaces.Dict(
                    {i: gym.spaces.Discrete(5) for i in range(21, 37)}
                ),
            }
        )
        # Action space is a tuple. First action: which dice you take. Second action: roll again or not.
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(6), gym.spaces.Discrete(2)))

    def _get_dice_sum(self) -> int:
        """Return the sum of the collected dices."""
        return_value = 0

        # Dice with eyes = number of eyes per die times the number of dice.
        for ind, die in enumerate(self._dice_collected):
            return_value += ind * die  # Worms have ind = 0 and hence are not counted here.

        # Worms have value five.
        return_value += self._dice_collected[0] * 5

        return return_value

    def _get_obs_dice(self):  # -> tuple[list[int], list[int]] or npt arrays if we keep them
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
        observation_tiles = (self.you, self.tile_table)
        return observation_tiles

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
            "self._get_sum()": self._get_dice_sum(),
            "self._get_obs_dice()": self._get_obs_dice(),
            "self._get_obs_tiles()": self._get_obs_tiles(),
            "self.legal_move(action)": self._legal_move(action),
        }
        return return_value

    def _soft_reset(self) -> None:
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
        self._dice_collected: list[int] = [0, 0, 0, 0, 0, 0]
        self._dice_rolled: list[int] = [0, 0, 0, 0, 0, 0]
        self.roll_counter = 0
        self.remaining_dice = 8
        self.tile_table = {
            21: 1,
            22: 1,
            23: 1,
            24: 1,
            25: 2,
            26: 2,
            27: 2,
            28: 2,
            29: 3,
            30: 3,
            31: 3,
            32: 3,
            33: 4,
            34: 4,
            35: 4,
            36: 4,
        }
        self.terminated = False
        self.truncated = False
        return_obs = {
            "dice_collected": np.array(self._dice_collected),
            "dice_rolled": np.array(self._dice_rolled),
            # "player": gym.spaces.Discrete(num_players),  # Number of players in the game.
            "tiles_table": self.tile_table,  # Tiles that can be taken.
            "tiles_player": self.you,
        }
        # self._dice_rolled = np.array([0, 2, 1, 4, 1, 0])

        for _ in range(self.num_dice):
            self._dice_rolled[rand.randint(0, 5)] += 1

        info = self._get_info((0, 1))  # Arbitrary action in reset only for debugging

        return return_obs, info

    def _legal_move(self, action: tuple[int, int]):
        """Check if action is allowed."""
        terminated: bool = False
        truncated: bool = False

        # TODO prove illegal moves
        # Dice already collected cannot be taken again.
        if self.roll_counter >= 2:
            self.terminated = True
            for index in range(len(self._dice_rolled)):
                if self._dice_rolled[index] > 0 and self._dice_collected[index] == 0:
                    self.terminated = False

            # for die_collected in self._dice_collected:
            #     for die_rolled in self._dice_rolled:
            #         if die_rolled > 0 and die_collected == 0:
            #             self.terminated = False

        if self.remaining_dice == 0 and self._get_dice_sum() < 21:
            self.terminated = True

        if self.remaining_dice == 0 and self._dice_collected[0] == 0:
            self.terminated = True

    def _step_dice(self, action: tuple[int, int]) -> None:
        """Execute one roll of the dice and picking or returning a tile.

        :param: action: The action to take: which dice to collect.
        """
        self._dice_collected[action[self.action_index_dice]] = self._dice_rolled[action[self.action_index_dice]]
        # Reduce the remaining number of dice by the number collected.
        self.remaining_dice -= self._dice_collected[action[self.action_index_dice]]
        self._dice_rolled: list[int] = [0, 0, 0, 0, 0, 0]
        self._legal_move(action)

        if self.terminated or self.truncated:
            return

        if action[self.action_index_roll] == self.action_roll:
            max_dice: int = self.num_dice - np.sum(self._dice_collected)
            dices_to_roll: int = min(self.remaining_dice, max_dice)
            for _ in range(dices_to_roll):
                self._dice_rolled[rand.randint(0, 5)] += 1
            self.roll_counter += 1
            self.truncated = False
        else:
            self.truncated = True

    def _step_tiles(self) -> int:
        """Internal step for picking or returning a tile."""
        return_value = 0
        dice_sum: int = self._get_dice_sum()

        # Environment takes the highest tile on the table.
        if dice_sum >= 21 and self.tile_table[dice_sum]:
            self.you[dice_sum] = self.tile_table[dice_sum]
            del self.tile_table[dice_sum]
            print("Your tiles:", self.you)
            return_value = self.you[dice_sum]
            self.truncated = True
            self._soft_reset()
        # Environment takes no Tile
        else:
            # Empty nothing happens, reward stays zero
            if not self.you:
                pass
            # You have at least one Tile and put it back to the Table
            else:
                moved_key, moved_value = self.you.popitem()
                self.tile_table[moved_key] = moved_value
                return_value = -moved_value
                # Moved Tile is highest
                if moved_key == max(self.tile_table):
                    pass
                # Tile not available to taken for the rest of the Game
                else:
                    del self.tile_table[moved_key]
        return return_value

    def step(self, action: tuple[int, int]):
        reward = 0
        self._legal_move(action)
        self._step_dice(action)
        if self.remaining_dice == 0 or action[self.action_index_roll] == self.action_stop:
            reward = self._step_tiles()

        return_obs = {
            "dice_collected": self._dice_collected,
            "dice_rolled": self._dice_rolled,
            # "player": gym.spaces.Discrete(num_players),  # Number of players in the game.
            "tiles_table": self.tile_table,  # Tiles that can be taken.
            "tiles_player": self.you,
        }
        info = self._get_info(action)

        return return_obs, reward, self.terminated, self.truncated, info


# The next 18 lines, until 'print(*line)', were copied from Stack Overflow


die_faces: list[str] = [
    "",  # index = 0 doesn't have a face
    "[     ]\n[  0  ]\n[     ]",  # index 1
    "[0    ]\n[     ]\n[    0]",  # index 2
    "[0    ]\n[  0  ]\n[    0]",  # index 3
    "[0   0]\n[     ]\n[0   0]",  # index 4
    "[0   0]\n[  0  ]\n[0   0]",  # index 5
    "[0   0]\n[0   0]\n[0   0]",  # index 6
]


def print_dice(values: list[int]) -> None:
    """Print one dice."""
    faces = [die_faces[v].splitlines() for v in values]
    for line in zip(*faces):
        print(*line)


def print_roll(observation, total) -> None:
    """Print one roll."""
    print_dice([1, 2, 3, 4, 5, 6])
    # Print line of collected dice.
    for collected in range(len(observation[0]) - 1):
        print(f"   {observation[0][collected + 1]}    ", end="")
    print(f"   {observation[0][0]}    ", end="")
    # Print sum at the end of the collected dice line
    print(f" collected   sum = {total}")
    # Print line of rolled dice.
    for rolled in range(len(observation[1]) - 1):
        print(f"   {observation[1][rolled + 1]}    ", end="")
    print(f"   {observation[1][0]}    ", end="")
    print(" rolled")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    NUMBER_OF_DICE: int = 8
    NUMBER_OF_PLAYERS: int = 2
    env = PickominoEnv(NUMBER_OF_PLAYERS)
    """Interactive test."""
    observation, info = env.reset()
    total = info["self._get_sum()"]
    dices_rolled_coll = observation["dice_collected"], observation["dice_rolled"]
    print("Reset")
    for step in range(6):
        print_roll(dices_rolled_coll, total)
        # action = (0, 1)  # dummy
        # print(f"act: {action}")
        # for key, value in info.items():
        #     print(key, value)
        # print("--------------------")
        selection: int = int(input("Which dice do you want to collect? (1..5 or worm =6) or -1 to stop: "))
        # print("step:", step, "    selection:", selection)
        if selection == -1:
            break
        if selection == 6:
            selection = 0  # Collecting a worm is the action (0, 1).
        action: tuple[int, int] = (selection, 0)
        observation, reward, terminated, truncated, info = env.step(action)
        dices_rolled_coll = observation["dice_collected"], observation["dice_rolled"]
        player_tiles = observation["tiles_player"]
        print(player_tiles)
