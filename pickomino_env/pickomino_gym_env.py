"""Pickomino game with gymnasium API."""

import random as rand
from typing import Any
import numpy as np
import gymnasium as gym
from numpy import ndarray, dtype


class PickominoEnv(gym.Env):
    """The environment class."""

    SMALLEST_TILE = 21
    LARGEST_TILE = 36
    ACTION_INDEX_DICE = 0
    ACTION_INDEX_ROLL = 1
    ACTION_ROLL = 0
    ACTION_STOP = 1

    def __init__(self, number_of_bots: int) -> None:
        """Constructor."""
        self._action: tuple[int, int] = 0, 0
        self._num_dice: int = 8
        self._roll_counter: int = 0
        self._number_of_bots: int = number_of_bots
        self._you: Player = Player(bot=False, name="You")
        self._players: list[Player] = []
        self._remaining_dice: int = self._num_dice
        self._terminated: bool = False
        self._truncated: bool = False
        self._no_throw = False

        self._dice = Dice()
        self._table_tiles = TableTiles()
        # Define what the agent can observe.
        # Dict space gives us structured, human-readable observations.
        # 6 possible faces of the dice. Worm = index 0, Rest: index = faces value of die
        # TODO you liste 0-16
        self.observation_space = gym.spaces.Dict(
            {
                "dice_collected": gym.spaces.Box(low=0, high=8, shape=(6,), dtype=np.int64),
                "dice_rolled": gym.spaces.Box(low=0, high=8, shape=(6,), dtype=np.int64),
                # Flatten tiles into a 16-length binary vector for SB3 compatibility (no nested Dict)
                "tiles_table": gym.spaces.Box(low=0, high=1, shape=(16,), dtype=np.int8),
                # 0 means no tile; 21..36 are valid tile ids
                "tile_players": gym.spaces.Discrete(self._create_players()),
            }
        )
        # Action space is a tuple. First action: which dice you take. Second action: roll again or not.
        self.action_space = gym.spaces.MultiDiscrete([6, 2])

    def _create_players(self) -> int:
        names = ["Alfa", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
        self._players.append(self._you)
        for i in range(self._number_of_bots):
            self._players.append(Player(bot=True, name=names[i]))

        return len(self._players)

    def _get_obs_dice(
        self,
    ) -> tuple[list[int], list[int]]:
        """Convert internal state to observation format.

        Returns:
            tuple: Dices collected and dices rolled.
        """

        return self._dice.get_collected(), self._dice.get_rolled()

    @staticmethod
    def get_worms(moved_key: int) -> int:
        """Give back the number of worms (1..4) for given the dice sum (21..36).
        Mapping:
        21–24 -> 1, 25–28 -> 2, 29–32 -> 3, 33–36 -> 4
        """
        if not PickominoEnv.SMALLEST_TILE <= moved_key <= PickominoEnv.LARGEST_TILE:
            raise ValueError("dice_sum must be between 21 and 36.")
        return (moved_key - PickominoEnv.SMALLEST_TILE) // 4 + 1

    def _get_obs_tiles(self) -> tuple[int, dict[int, bool]]:
        """Convert internal state to observation format.

        Returns:
            dict: Tiles distribution
        """
        return self._you.show(), self._table_tiles.get_table()

    def _tiles_vector(self) -> ndarray[Any, dtype[Any]]:
        """Return tiles_table as a flat binary vector of length 16 for indices 21..36."""
        return np.array([1 if self._table_tiles.get_table()[i] else 0 for i in range(21, 37)], dtype=np.int8)

    def _current_obs(self) -> dict[str, ndarray[Any, dtype[Any]] | list[int]]:
        return {
            "dice_collected": np.array(self._dice.get_collected()),
            "dice_rolled": np.array(self._dice.get_rolled()),
            "tiles_table": self._tiles_vector(),
            "tile_players": list(map(lambda p: p.show(), self._players)),
        }

    def _get_info(self) -> dict[str, object]:
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with additional information which is useful for debugging but not necessary for learning.
        """
        return_value = {
            "action": self._action,
            "num_dice": self._num_dice,
            "remaining_dice": self._remaining_dice,
            "dice_collected": self._dice.get_collected(),
            "dice_rolled": self._dice.get_rolled(),
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "sum": self._dice.score()[0],
            "terminated": self._terminated,
            "get_obs_dice()": self._get_obs_dice(),
            "get_obs_tiles()": self._get_obs_tiles(),
            "tiles_table_vec": self._tiles_vector(),
            "dice": self._dice,
            # "self.legal_move(action)": self._legal_move(action),
        }
        return return_value

    def _soft_reset(self) -> None:
        """Clear collected and rolled and roll again"""

        self._dice = Dice()
        self._no_throw = False
        # print(f"PRINT DEBUGGING - rolling {self._num_dice} dice.")
        self._dice.roll()

    def _remove_tile_from_player(self) -> int:
        return_value = 0

        if self._you.show():
            tile_to_return: int = self._you.remove_tile()  # Remove the tile from the player.
            # print("PRINT DEBUGGING - Returning tile:", tile_to_return, "to the table.")
            self._table_tiles.get_table()[tile_to_return] = True  # Return the tile to the table.
            return_value = -self.get_worms(tile_to_return)  # Reward is MINUS the value of the returned tile.
            # If the returned tile is not the highest, turn the highest tile around (set to False)
            # Search for the highest tile to turn.
            highest = self._table_tiles.highest()
            # Turn the highest tile if there is one.
            if highest:
                self._table_tiles.set_tile(highest, False)

        return return_value

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        self._dice = Dice()
        self._you = Player(bot=False, name="You")
        self._table_tiles = TableTiles()
        self._no_throw = False
        self._terminated = False
        self._truncated = False

        # print(f"PRINT DEBUGGING - rolling {self._num_dice} dice.")
        self._dice.roll()

        return_obs = self._current_obs()

        return return_obs, self._get_info()

    def _legal_move(self) -> bool:
        """Check if action is allowed."""
        self._terminated = False
        self._truncated = False
        return_value: bool = True

        if self._action[PickominoEnv.ACTION_INDEX_DICE] not in range(1, 6):
            self._terminated = True
        else:
            # Dice already collected cannot be taken again.
            if self._dice.get_collected()[self._action[PickominoEnv.ACTION_INDEX_DICE]] != 0:
                if self._dice.get_rolled()[self._action[PickominoEnv.ACTION_INDEX_DICE]] != 0:
                    self._terminated = True

            # Action if no dice is available in rolled_dice
            if not self._dice.get_rolled()[self._action[PickominoEnv.ACTION_INDEX_DICE]]:
                self._terminated = True

            # No dice left and 21 not reached
            if self._remaining_dice == 0 and self._dice.score()[0] < PickominoEnv.SMALLEST_TILE:
                self._terminated = True

            # No worm collected
            if self._remaining_dice == 0 and not self._dice.score()[1]:
                self._terminated = True

        if self._terminated:
            return_value = False

        return return_value

    def _step_dice(self) -> None:
        """Execute one roll of the dice and picking or returning a tile.

        :param: action: The action to take: which dice to collect.
        """
        # Check legal move before adding selected dice to be collected.
        if self._legal_move():
            self._dice.collect(self._action[PickominoEnv.ACTION_INDEX_DICE])
        else:
            return

        if self._truncated:
            return

        # Action is to roll
        if self._action[PickominoEnv.ACTION_INDEX_ROLL] == PickominoEnv.ACTION_ROLL:
            self._dice.roll()
            # Check for no-throw
            # TODO Check the no throw concept.
            self._no_throw = True
            for index in range(len(self._dice.get_rolled())):
                if self._dice.get_rolled()[index] > 0 and self._dice.get_collected()[index] == 0:
                    self._no_throw = False
            # if self._no_throw:
            #     # print(f"PRINT DEBUGGING - no-throw.")
            self._truncated = False

        # Action is to stop rolling dice and pick a tile.
        else:
            self._truncated = True
            if self._dice.get_collected()[5] == 0:
                self._terminated = True
                self._no_throw = True

    def _step_tiles(self) -> int:
        """Pick or return a tile.

        Internal sub-step for picking or returning a tile after finishing rolling dice.

        :return: Value of moving the tile [-4 ... +4]
        """
        dice_sum: int = self._dice.score()[0]
        # print("PRINT DEBUGGING - dice_sum: ", dice_sum)

        # Using dice_sum as an index in [21..36] below, hence for dice_sum < 21 need to return early.
        # No throw or 21 not reached -> return tile
        if self._no_throw or dice_sum < PickominoEnv.SMALLEST_TILE:
            return_value = self._remove_tile_from_player()
            # print("PRINT DEBUGGING - Turning tile:", highest, "on the table.")
            # print("PRINT DEBUGGING - Your tiles:", self.you)
            self._soft_reset()
            return return_value

        # Using dice_sum as an index in [21..36], higher rolls can only pick 36 or lower
        dice_sum = min(dice_sum, PickominoEnv.LARGEST_TILE)
        # Environment takes the highest tile on the table.
        # Only pick a tile if it is on the table.
        if self._table_tiles.get_table()[dice_sum]:
            # print("PRINT DEBUGGING - Picking tile:", dice_sum)
            self._you.add_tile(dice_sum)  # Add the tile to the player.
            self._table_tiles.set_tile(dice_sum, False)  # Mark the tile as no longer on the table.
            return_value = self.get_worms(dice_sum)
            self._truncated = True
        # Tile is not available on the table
        else:
            # Pick the highest of the tiles smaller than the unavailable tile
            # Find the highest tile smaller than the dice sum.
            highest = self._table_tiles.highest()
            if highest:  # Found the highest tile to pick from the table.
                # print("PRINT DEBUGGING - Picking tile:", highest)
                self._you.add_tile(highest)  # Add the tile to the player.
                self._table_tiles.set_tile(highest, False)  # Mark the tile as no longer on the table.
                return_value = self.get_worms(highest)
                self._truncated = True
            # Also no smaller tiles available -> have to return players showing tile if there is one.
            else:
                return_value = self._remove_tile_from_player()
                # print("PRINT DEBUGGING - Turning tile:", highest, "on the table.")

        # print("PRINT DEBUGGING - Your tiles:", self.you)
        self._soft_reset()
        return return_value

    def step(self, action: tuple[int, int]) -> tuple[dict[str, Any], int, bool, bool, dict[str, object]]:
        self._action = action
        reward = 0
        self._legal_move()
        self._step_dice()
        if (
            self._remaining_dice == 0
            or self._action[PickominoEnv.ACTION_INDEX_ROLL] == PickominoEnv.ACTION_STOP
            or self._no_throw
        ):
            reward = self._step_tiles()

            # Game Over if no Tile is on the table anymore.
            if not self._table_tiles.highest():
                self._terminated = True

        if self._terminated:
            return self._current_obs(), reward, self._terminated, self._truncated, self._get_info()

        return_obs = self._current_obs()

        info = self._get_info()

        return return_obs, reward, self._terminated, self._truncated, info


# The next 18 lines, until 'print(*line)', were copied from Stack Overflow


class Dice:
    """Represents a collection of die face frequencies.
    An example for eight dice with six faces is: (0, 0, 3, 4, 0, 1)
    This example means that three threes, four fours and one worm die face have been collected."""

    def __init__(self) -> None:
        self._values: list[int] = [1, 2, 3, 4, 5, 5]  # Worm has the value 5.
        self._n_dice: int = 8
        self._collected: list[int] = [0, 0, 0, 0, 0, 0]  # Collected dice, up to 8 per side.
        self._rolled: list[int] = [0, 0, 0, 0, 0, 0]  # Last roll.

    def collect(self, action_face: int) -> list[int]:
        """Insert chosen face to collected list."""
        self._collected[action_face] = self._rolled[action_face]
        return self._collected

    def get_collected(self) -> list[int]:
        """Getter."""
        return self._collected

    def get_rolled(self) -> list[int]:
        """Getter."""
        return self._rolled

    def roll(self) -> list[int]:
        """Roll remaining dice."""
        self._rolled = [0, 0, 0, 0, 0, 0]

        for _ in range(self._n_dice - sum(self._collected)):
            self._rolled[rand.randint(0, 5)] += 1

        return self._rolled

    def score(self) -> tuple[int, bool]:
        """Count the score based on an array of die face frequencies."""
        # Check if there is at least one worm
        has_worms = self._collected[-1] > 0
        # Multiply the frequency of each die face with its value

        current_score = int(np.dot(self._values, self._collected) if self._collected else 0)

        return current_score, has_worms

    def __str__(self) -> str:
        die_faces: list[str] = [
            "",  # index = 0 doesn't have a face
            "[     ]\n[  0  ]\n[     ]",  # index 1
            "[0    ]\n[     ]\n[    0]",  # index 2
            "[0    ]\n[  0  ]\n[    0]",  # index 3
            "[0   0]\n[     ]\n[0   0]",  # index 4
            "[0   0]\n[  0  ]\n[0   0]",  # index 5
            "[0   0]\n[0   0]\n[0   0]",  # index 6
        ]
        # Print one dice.
        show_values = [1, 2, 3, 4, 5, 6]
        faces = [die_faces[v].splitlines() for v in show_values]
        return_value = ""
        for line in zip(*faces):
            return_value += " ".join(line) + "\n"
        return return_value


class TableTiles:
    """Define the Tiles on the Table"""

    def __init__(self) -> None:
        """Constructor"""
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

    def set_tile(self, tile_number: int, truth_value: bool) -> None:
        """Set one Tile."""
        self._tile_table[tile_number] = truth_value

    def get_table(self) -> dict[int, bool]:
        """Get whole table."""
        return self._tile_table

    def is_empty(self) -> bool:
        """Check if the table is empty"""
        if self._tile_table.values():
            return False
        return True

    def highest(self) -> int:
        """Get the highest tile on table"""
        highest = 0
        if not self.is_empty():
            for key, value in self._tile_table.items():
                if value:
                    highest = key
        return highest


class Player:
    """Player class with his tiles and name."""

    def __init__(self, bot: bool, name: str) -> None:
        """Constructor for a player"""
        self.name: str = name
        self.tile_stack: list[int] = []
        self.bot: bool = bot

    def show(self) -> int:
        """Show the tile from the player stack."""
        if self.tile_stack:
            return self.tile_stack[-1]
        return 0

    def add_tile(self, tile: int) -> None:
        """Add tile to the player stack"""
        self.tile_stack.append(tile)

    def remove_tile(self) -> int:
        """Remove the top tile from the player stack."""
        return self.tile_stack.pop()

    def score(self) -> int:
        """Return player score at the end of game"""
        score: int = 0
        for tile in self.tile_stack:
            score += PickominoEnv.get_worms(tile)
        return score


def print_roll(observation: tuple[list[int], list[int]], total: int, dice: object) -> None:
    """Print one roll."""
    print(dice)
    # Print line of collected dice.
    for collected in range(len(observation[0])):
        print(f"   {observation[0][collected]}    ", end="")
    # Print sum at the end of the collected dice line
    print(f" collected   sum = {total}")
    # Print line of rolled dice.
    for rolled in range(len(observation[1])):
        print(f"   {observation[1][rolled]}    ", end="")
    print(" rolled")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    # Interactive test.
    # TODO: not yet used.
    # NUMBER_OF_DICE: int = 8
    # NUMBER_OF_PLAYERS: int = 2
    MAX_TURNS: int = 300
    env = PickominoEnv(2)
    game_observation, game_info = env.reset()
    game_reward: int = 0
    # for key, value in info.items():
    #     print(key, value)
    GAME_TOTAL = game_info["sum"]
    dice_rolled_coll = game_observation["dice_collected"], game_observation["dice_rolled"]
    print("Reset")
    for step in range(MAX_TURNS):
        print("Step:", step)
        print("Your showing tile: ", game_observation["tile_players"], "(your reward = ", game_reward, ")")
        print_roll(dice_rolled_coll, GAME_TOTAL, game_info["dice"])
        print("Tiles on table:", end=" ")
        for ind, game_tile in enumerate(game_observation["tiles_table"]):
            if game_tile:
                print(ind + 21, end=" ")
        print()
        SELECTION: int = int(input("Which dice do you want to collect? (1..5 or worm =6): ")) - 1
        stop: int = int(input("Keep rolling? (0 = ROLL,  1 = STOP: "))
        print()
        game_action = (SELECTION, stop)
        game_observation, game_reward, game_terminated, game_truncated, game_info = env.step(game_action)
        dice_rolled_coll = game_observation["dice_collected"], game_observation["dice_rolled"]
        GAME_TOTAL = game_info["sum"]
        print(game_terminated, game_truncated)
