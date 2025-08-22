"""Pickomino game with gymnasium API."""

import random as rand
from typing import Optional
import numpy as np
import gymnasium as gym


class PickominoEnv(gym.Env):
    """The environment class."""

    def __init__(self) -> None:
        """Constructor."""
        self._action_index_dice: int = 0
        self._action_index_roll: int = 1
        self._action_roll: int = 0
        self._action_stop: int = 1
        self._num_dice: int = 8
        self._roll_counter: int = 0
        self._remaining_dice: int = self._num_dice
        self._terminated: bool = False
        self._truncated: bool = False
        self._no_throw = False
        self._you: list[int] = []
        # Define what the agent can observe.
        # Dict space gives us structured, human-readable observations.
        # 6 possible faces of the dice. Worm = index 0, Rest: index = faces value of die
        self._dice_collected: list[int] = [0, 0, 0, 0, 0, 0]  # Collected dice, up to 8 per side.
        self._dice_rolled: list[int] = [0, 0, 0, 0, 0, 0]  # Last roll.

        self._tile_table: dict[int, bool] = {
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

        self.observation_space = gym.spaces.Dict(
            {
                "dice_collected": gym.spaces.Box(low=0, high=8, shape=(6,), dtype=np.int64),
                "dice_rolled": gym.spaces.Box(low=0, high=6, shape=(6,), dtype=np.int64),
                # Flatten tiles into a 16-length binary vector for SB3 compatibility (no nested Dict)
                "tiles_table": gym.spaces.Box(low=0, high=1, shape=(16,), dtype=np.int8),
                # 0 means no tile; 21..36 are valid tile ids
                "tile_player": gym.spaces.Discrete(37),
            }
        )

        # Action space is a tuple. First action: which dice you take. Second action: roll again or not.
        self.action_space = gym.spaces.MultiDiscrete((6, 2))

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
        observation_tiles = (self._you, self._tile_table)
        return observation_tiles

    def _tiles_vector(self) -> np.ndarray:
        """Return tiles_table as a flat binary vector of length 16 for indices 21..36."""
        return np.array([1 if self._tile_table[i] else 0 for i in range(21, 37)], dtype=np.int8)

    def _current_obs(self):
        return {
            "dice_collected": np.array(self._dice_collected),
            "dice_rolled": np.array(self._dice_rolled),
            "tiles_table": self._tiles_vector(),
            "tile_player": (self._you[-1] if self._you else 0),
        }

    def _get_info(self, action):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with additional information which is useful for debugging but not necessary for learning.
        """
        return_value = {
            "action": action,
            "num_dice": self._num_dice,
            "remaining_dice": self._remaining_dice,
            "dice_collected": self._dice_collected,
            "dice_rolled": self._dice_rolled,
            "roll_counter": self._roll_counter,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "sum": self._get_dice_sum(),
            "terminated": self._terminated,
            "get_obs_dice()": self._get_obs_dice(),
            "get_obs_tiles()": self._get_obs_tiles(),
            "tiles_table_vec": self._tiles_vector(),
            # "self.legal_move(action)": self._legal_move(action),
        }
        return return_value

    def _soft_reset(self) -> None:
        self._dice_collected: list[int] = [0, 0, 0, 0, 0, 0]
        self._dice_rolled: list[int] = [0, 0, 0, 0, 0, 0]
        self._roll_counter = 0
        self._remaining_dice = 8
        self._no_throw = False
        # print(f"PRINT DEBUGGING - rolling {self._num_dice} dice.")
        for _ in range(self._num_dice):
            self._dice_rolled[rand.randint(0, 5)] += 1

    def _remove_tile_from_player(self) -> int:
        return_value = 0
        if self._you:
            tile_to_return: int = self._you.pop()  # Remove the tile from the player.
            # print("PRINT DEBUGGING - Returning tile:", tile_to_return, "to the table.")
            self._tile_table[tile_to_return] = True  # Return the tile to the table.
            return_value = -self._get_worms(tile_to_return)  # Reward is MINUS the value of the returned tile.
            # If the returned tile is not the highest, turn the highest tile around (set to False)
            # Search for the highest tile to turn.
            highest = 0
            for tile in range(tile_to_return + 1, 37):
                if self._tile_table[tile]:
                    highest = tile
            # Turn the highest tile if there is one.
            if highest:
                self._tile_table[highest] = False

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
        self._dice_collected: list[int] = [0, 0, 0, 0, 0, 0]
        self._dice_rolled: list[int] = [0, 0, 0, 0, 0, 0]
        self._roll_counter = 0
        self._remaining_dice = 8
        self._no_throw = False
        self._you = []
        self._tile_table = {
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
        self._terminated = False
        self._truncated = False

        # print(f"PRINT DEBUGGING - rolling {self._num_dice} dice.")
        for _ in range(self._num_dice):
            self._dice_rolled[rand.randint(0, 5)] += 1

        return_obs = self._current_obs()

        info = self._get_info((0, 1))  # Arbitrary action in reset only for debugging

        return return_obs, info

    def _legal_move(self, action: tuple[int, int]):
        """Check if action is allowed."""
        self._terminated: bool = False
        self._truncated: bool = False
        return_value: bool = True
        # Dice already collected cannot be taken again.
        if not self._dice_collected[action[self._action_index_dice]] == 0:
            if not self._dice_rolled[action[self._action_index_dice]] == 0:
                self._terminated = True

        # Action if no dice is available in rolled_dice
        if not self._dice_rolled[action[self._action_index_dice]]:
            self._terminated = True

        # No dice left and 21 not reached
        if self._remaining_dice == 0 and self._get_dice_sum() < 21:
            self._terminated = True

        # No worm collected
        if self._remaining_dice == 0 and self._dice_collected[0] == 0:
            self._terminated = True

        if self._terminated:
            return_value = False

        return return_value

    def _roll(self) -> None:
        max_dice: int = self._num_dice - np.sum(self._dice_collected)
        dices_to_roll: int = min(self._remaining_dice, max_dice)
        for _ in range(dices_to_roll):
            self._dice_rolled[rand.randint(0, 5)] += 1
        self._roll_counter += 1
        self._truncated = False

    def _step_dice(self, action: tuple[int, int]) -> None:
        """Execute one roll of the dice and picking or returning a tile.

        :param: action: The action to take: which dice to collect.
        """
        # Check legal move before adding selected dice to be collected.
        if self._legal_move(action):
            self._dice_collected[action[self._action_index_dice]] = self._dice_rolled[action[self._action_index_dice]]
        else:
            return
        # Reduce the remaining number of dice by the number collected.
        self._remaining_dice -= self._dice_collected[action[self._action_index_dice]]
        # Reset diced rolled before rolling again below.
        self._dice_rolled = [0, 0, 0, 0, 0, 0]

        if self._truncated:
            return

        # Action is to roll
        if action[self._action_index_roll] == self._action_roll:
            # TODO: Jarl: we have updated self._remaining_dice, why do we need to do this?
            max_dice: int = self._num_dice - np.sum(self._dice_collected)
            dices_to_roll: int = min(self._remaining_dice, max_dice)
            # print(f"PRINT DEBUGGING - rolling {dices_to_roll} dice.")
            for _ in range(dices_to_roll):
                self._dice_rolled[rand.randint(0, 5)] += 1
            # Check for no-throw
            self._no_throw = True
            for index in range(len(self._dice_rolled)):
                if self._dice_rolled[index] > 0 and self._dice_collected[index] == 0:
                    self._no_throw = False
            # if self._no_throw:
            #     # print(f"PRINT DEBUGGING - no-throw.")
            self._roll_counter += 1
            self._truncated = False

        # Action is to stop rolling dice and pick a tile.
        else:
            self._truncated = True
            if self._dice_collected[0] == 0:
                self._terminated = True
                self._no_throw = True

    def _get_worms(self, moved_key: int) -> int:
        """Give back the number of worms (1..4) for given the dice sum (21..36).
        Mapping:
        21–24 -> 1, 25–28 -> 2, 29–32 -> 3, 33–36 -> 4
        """
        if not 21 <= moved_key <= 36:
            raise ValueError("dice_sum must be between 21 and 36.")
        return (moved_key - 21) // 4 + 1

    def _step_tiles(self) -> int:
        """Pick or return a tile.

        Internal sub-step for picking or returning a tile after finishing rolling dice.

        :return: Value of moving the tile [-4 ... +4]
        """
        dice_sum: int = self._get_dice_sum()
        # print("PRINT DEBUGGING - dice_sum: ", dice_sum)

        # Using dice_sum as an index in [21..36] below, hence for dice_sum < 21 need to return early.
        # No throw or 21 not reached -> return tile
        if self._no_throw or dice_sum < 21:
            return_value = self._remove_tile_from_player()
            # print("PRINT DEBUGGING - Turning tile:", highest, "on the table.")
            # print("PRINT DEBUGGING - Your tiles:", self.you)
            self._soft_reset()
            return return_value

        # Using dice_sum as an index in [21..36], higher rolls can only pick 36 or lower
        if dice_sum > 36:
            dice_sum = 36
        # Environment takes the highest tile on the table.
        # Only pick a tile if it is on the table.
        if self._tile_table[dice_sum]:
            # print("PRINT DEBUGGING - Picking tile:", dice_sum)
            self._you.append(dice_sum)  # Add the tile to the player.
            self._tile_table[dice_sum] = False  # Mark the tile as no longer on the table.
            return_value = self._get_worms(dice_sum)
            self._truncated = True
        # Tile is not available on the table
        else:
            # Pick the highest of the tiles smaller than the unavailable tile
            # Find the highest tile smaller than the dice sum.
            highest = 0
            for tile in range(21, dice_sum):
                if self._tile_table[tile]:
                    highest = tile
            if highest:  # Found the highest tile to pick from the table.
                # print("PRINT DEBUGGING - Picking tile:", highest)
                self._you.append(highest)  # Add the tile to the player.
                self._tile_table[highest] = False  # Mark the tile as no longer on the table.
                return_value = self._get_worms(highest)
                self._truncated = True
            # Also no smaller tiles available -> have to return players showing tile if there is one.
            else:
                return_value = self._remove_tile_from_player()
                # print("PRINT DEBUGGING - Turning tile:", highest, "on the table.")

        # print("PRINT DEBUGGING - Your tiles:", self.you)
        self._soft_reset()
        return return_value

    def step(self, action: tuple[int, int]):
        reward = 0
        self._legal_move(action)
        self._step_dice(action)
        if self._remaining_dice == 0 or action[self._action_index_roll] == self._action_stop or self._no_throw:
            reward = self._step_tiles()

            # Game Over if no Tile is on the table anymore.
            self._terminated = True
            for _, value in self._tile_table.items():
                if value:
                    self._terminated = False

        if self._terminated:
            return self._current_obs(), reward, self._terminated, self._truncated, self._get_info(action)

        return_obs = self._current_obs()

        info = self._get_info(action)

        return return_obs, reward, self._terminated, self._truncated, info


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
    # Interactive test.
    # TODO: not yet used.
    # NUMBER_OF_DICE: int = 8
    # NUMBER_OF_PLAYERS: int = 2
    MAX_TURNS: int = 300
    env = PickominoEnv()
    observation, info = env.reset()
    reward: int = 0
    # for key, value in info.items():
    #     print(key, value)
    total: int = info["self._sum"]
    dices_rolled_coll = observation["dice_collected"], observation["dice_rolled"]
    print("Reset")
    for step in range(MAX_TURNS):
        print("Step:", step)
        print("Your showing tile: ", observation["tile_player"], "(your reward = ", reward, ")")
        print_roll(dices_rolled_coll, total)
        print("Tiles on table:", end=" ")
        for tile in observation["tiles_table"]:
            if observation["tiles_table"][tile]:
                print(tile, end=" ")
        print()
        selection: int = int(input("Which dice do you want to collect? (1..5 or worm =6): "))
        stop: int = int(input("Keep rolling? (0 = ROLL,  1 = STOP: "))
        print()
        if selection == 6:
            selection = 0  # Collecting a worm internally has index 0.
        action: tuple[int, int] = (selection, stop)
        observation, reward, terminated, truncated, info = env.step(action)
        dices_rolled_coll = observation["dice_collected"], observation["dice_rolled"]
        total = info["self._sum"]
        print(terminated)
