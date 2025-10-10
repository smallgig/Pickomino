"""Pickomino game with gymnasium API."""

from typing import Any
import numpy as np
import gymnasium as gym
from numpy import ndarray, dtype

from pickomino_env.src.dice import Dice
from pickomino_env.src.table_tiles import TableTiles
from pickomino_env.src.player import Player
from pickomino_env.src import utils

RED = "\033[31m"
NO_RED = "\033[0m"
GREEN = "\033[32m"
NO_GREEN = "\033[0m"


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
        self._failed_attempt: bool = False
        self._explanation: str = "Constructor"  # Reason for terminated, truncated or failed attempt.

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

    def _get_obs_tiles(self) -> tuple[int, dict[int, bool]]:
        """Convert internal state to observation format.

        Returns:
            dict: Tiles distribution
        """
        return self._players[0].show(), self._table_tiles.get_table()

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
            "terminated_reason": self._explanation,
            "dice": self._dice,
            "explanation": self._explanation,
            "failed_attempt": self._failed_attempt,
            "player_stack": self._players[0].show_all(),
            "smallest_tile": self._table_tiles.smallest(),
            # "self.legal_move(action)": self._legal_move(action),
        }
        return return_value

    def _soft_reset(self) -> None:
        """Clear collected and rolled and roll again"""

        self._dice = Dice()
        self._failed_attempt = False
        self._roll_counter = 0
        # print(f"PRINT DEBUGGING - rolling {self._num_dice} dice.")
        self._dice.roll()

    def _remove_tile_from_player(self) -> int:
        return_value = 0

        if self._players[0].show():
            tile_to_return: int = self._players[0].remove_tile()  # Remove the tile from the player.
            # print("PRINT DEBUGGING - Returning tile:", tile_to_return, "to the table.")
            self._table_tiles.get_table()[tile_to_return] = True  # Return the tile to the table.
            return_value = -utils.get_worms(tile_to_return)  # Reward is MINUS the value of the returned tile.
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
        self._failed_attempt = False
        self._terminated = False
        self._truncated = False

        # print(f"PRINT DEBUGGING - rolling {self._num_dice} dice.")
        self._dice.roll()

        return_obs = self._current_obs()

        return return_obs, self._get_info()

    def _action_is_allowed(self) -> None:
        """Check if action is allowed."""
        self._terminated = False
        self._truncated = False
        self._failed_attempt = False

        # Check action values are within range
        if self._action[PickominoEnv.ACTION_INDEX_DICE] not in range(0, 6) or self._action[
            PickominoEnv.ACTION_INDEX_ROLL
        ] not in range(0, 2):
            self._terminated = True
            self._explanation = RED + "Terminated: Action index not in range" + NO_RED
        # Selected Face value not rolled.
        if self._dice.get_rolled()[self._action[PickominoEnv.ACTION_INDEX_DICE]] == 0:
            self._truncated = True
            self._explanation = RED + "Truncated: Selected Face value not rolled" + NO_RED

        # Dice already collected cannot be taken again.
        if self._dice.get_collected()[self._action[PickominoEnv.ACTION_INDEX_DICE]] != 0:
            self._truncated = True
            self._explanation = RED + "Truncated: Dice already collected cannot be taken again" + NO_RED

        remaining_dice = self._dice.get_rolled().copy()
        remaining_dice[self._action[PickominoEnv.ACTION_INDEX_DICE]] = 0

        if self._action[PickominoEnv.ACTION_INDEX_ROLL] == PickominoEnv.ACTION_ROLL and not remaining_dice:
            self._truncated = True
            self._explanation = RED + "Truncated: No Dice left to roll and roll action selected." + NO_RED

        # Action allowed tryed to take tile

    def _step_dice(self) -> None:
        """Execute one roll of the dice and picking or returning a tile.

        :param: action: The action to take: which dice to collect.
        """
        self._set_failed_already_collected()

        self._dice.collect(self._action[PickominoEnv.ACTION_INDEX_DICE])

        self._set_failed_to_low()
        self._set_failed_no_worms()

        # Action is to roll
        if self._action[PickominoEnv.ACTION_INDEX_ROLL] == PickominoEnv.ACTION_ROLL:
            self._dice.roll()
            self._set_failed_already_collected()  # TODO: Is this needed?
            self._set_failed_to_low()
            self._set_failed_no_worms()

    def _step_tiles(self) -> int:
        """Pick or return a tile.

        Internal sub-step for picking or returning a tile after finishing rolling dice.

        :return: Value of moving the tile [-4 ... +4]
        """
        dice_sum: int = self._dice.score()[0]
        # print("PRINT DEBUGGING - dice_sum: ", dice_sum)

        # Using dice_sum as an index in [21..36] below, hence for dice_sum < 21 need to return early.
        # No throw or 21 not reached -> return tile
        if self._failed_attempt:
            return_value = self._remove_tile_from_player()
            # print("PRINT DEBUGGING - Turning tile:", highest, "on the table.")
            # print("PRINT DEBUGGING - Your tiles:", self.you)
            self._soft_reset()
            return return_value
        # Environment takes the highest tile on the table.
        # Only pick a tile if it is on the table.
        if self._table_tiles.get_table()[dice_sum]:
            # print("PRINT DEBUGGING - Picking tile:", dice_sum)
            self._players[0].add_tile(dice_sum)  # Add the tile to the player.
            self._table_tiles.set_tile(dice_sum, False)  # Mark the tile as no longer on the table.
            return_value = utils.get_worms(dice_sum)
        # Tile is not available on the table
        else:
            # Pick the highest of the tiles smaller than the unavailable tile
            # Find the highest tile smaller than the dice sum.
            highest: int = self._table_tiles.find_next_lower_tile(dice_sum)
            if highest:  # Found the highest tile to pick from the table.
                # print("PRINT DEBUGGING - Picking tile:", highest)
                self._players[0].add_tile(highest)  # Add the tile to the player.
                self._table_tiles.set_tile(highest, False)  # Mark the tile as no longer on the table.
                return_value = utils.get_worms(highest)
            # Also no smaller tiles available -> have to return players showing tile if there is one.
            else:
                return_value = self._remove_tile_from_player()
                self._explanation = RED + "No available tile on the table to take" + NO_RED
                # print("PRINT DEBUGGING - Turning tile:", highest, "on the table.")

        # print("PRINT DEBUGGING - Your tiles:", self.you)
        self._soft_reset()
        return return_value

    def _set_failed_already_collected(self):
        """Check if a dice is available to take"""
        can_take = any(
            rolled > 0 and collected == 0
            for rolled, collected in zip(self._dice.get_rolled(), self._dice.get_collected())
        )

        self._failed_attempt = not can_take
        self._explanation = (
            GREEN + "Good case" + NO_GREEN
            if can_take
            else RED + f"Failed: Collected was {self._dice.get_collected()}\n"
            f"No possible rolled dice to taken in {self._dice.get_rolled()}" + NO_RED
        )

    def _set_failed_to_low(self):
        """Failed: 21 not reached and action stop or no dice left"""
        if self._dice.score()[0] < PickominoEnv.SMALLEST_TILE:
            if self._action[PickominoEnv.ACTION_INDEX_ROLL] == PickominoEnv.ACTION_STOP:
                self._failed_attempt = True
                self._explanation = RED + "Failed: 21 not reached and action stop" + NO_RED

            if sum(self._dice.get_collected()) == 8 and self._dice.score()[0] < PickominoEnv.SMALLEST_TILE:
                self._failed_attempt = True
                self._explanation = RED + "Failed: 21 not reached and no dice left" + NO_RED

    def _set_failed_no_worms(self):
        """No worm collected and action stop"""
        if not self._dice.score()[1] and self._action[PickominoEnv.ACTION_INDEX_ROLL] == PickominoEnv.ACTION_STOP:
            self._failed_attempt = True
            self._explanation = RED + "Failed: No worm collected" + NO_RED

    def step(self, action: tuple[int, int]) -> tuple[dict[str, Any], int, bool, bool, dict[str, object]]:
        self._action = action
        reward = 0
        # Check legal move before doing a step.
        self._action_is_allowed()

        # TODO in one line like truncated
        if self._terminated:
            obs, reward, terminated, truncated, info = (
                self._current_obs(),
                reward,
                self._terminated,
                self._truncated,
                self._get_info(),
            )
            self.reset()
            return obs, reward, terminated, truncated, info
        if self._truncated:
            return self._current_obs(), reward, self._terminated, self._truncated, self._get_info()

        # Collect and roll the dice
        self._step_dice()

        # Action is to stop or failed attempt, get reward from step tiles.
        if self._action[PickominoEnv.ACTION_INDEX_ROLL] == PickominoEnv.ACTION_STOP or self._failed_attempt:
            # self._set_failed_attempt()
            reward = self._step_tiles()

        # Game Over if no Tile is on the table anymore.
        if not self._table_tiles.highest():
            self._terminated = True
            self._explanation = GREEN + "No Tile on the table, GAME OVER!" + NO_GREEN

        return self._current_obs(), reward, self._terminated, self._truncated, self._get_info()


def print_roll(observation: tuple[list[int], list[int]], total: int, dice: object) -> None:
    """Print one roll."""
    print(dice)
    # Print line of collected dice.
    for collected in range(len(observation[0])):
        print(f"   {observation[0][collected]}      ", end="")
    # Print sum at the end of the collected dice line
    print(f" collected   sum = {total}")
    # Print line of rolled dice.
    for rolled in range(len(observation[1])):
        print(f"   {observation[1][rolled]}      ", end="")
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
        explanation = game_info["explanation"]
        print(f"Terminated: {game_terminated} Truncated:{game_truncated} \nExplanation: {explanation}")
        print(f"Rolled: {game_observation['dice_rolled']}")
        if game_terminated:
            game_observation, game_info = env.reset()
