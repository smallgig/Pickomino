"""Pickomino game with gymnasium API."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from numpy import dtype, ndarray

from pickomino_env.src.bot import Bot
from pickomino_env.src.checker import Checker
from pickomino_env.src.constants import (  # Coloured printouts, game and action constants.
    ACTION_INDEX_DICE,
    ACTION_INDEX_ROLL,
    ACTION_ROLL,
    ACTION_STOP,
    GREEN,
    LARGEST_TILE,
    NO_GREEN,
    NO_RED,
    NUM_DICE,
    RED,
    SMALLEST_TILE,
)
from pickomino_env.src.dice import Dice
from pickomino_env.src.player import Player
from pickomino_env.src.table_tiles import TableTiles


class PickominoEnv(gym.Env):  # type: ignore[type-arg] # pylint: disable=too-many-instance-attributes.
    """The environment class."""

    def __init__(self, number_of_bots: int) -> None:
        """Construct the environment."""
        # The following is an idea for refactoring.
        # Have only on complex variable with the return value of the step function.
        self._action: tuple[int, int] = 0, 0  # Candidate for class Checker.
        self._number_of_bots: int = number_of_bots  # Remove this and use len(self._players) - 1 instead.
        self._you: Player = Player(bot=False, name="You")  # Put this in the players list and remove it from here.
        self._players: list[Player] = []
        self._terminated: bool = False
        self._truncated: bool = False
        self._failed_attempt: bool = False  # Candidate for class Checker.
        self._explanation: str = "Constructor"  # The reason, why the terminated, truncated or failed attempt is set.
        self._current_player_index: int = 0  # 0 for the player, 1 or more for bots.
        self._dice: Dice = Dice()
        self._table_tiles: TableTiles = (
            TableTiles()
        )  # Consider a complex class Table consisting of table tiles and players tiles.
        self._checker: Checker = Checker(self._dice, self._players, self._table_tiles)
        self._create_players()  # Do not move this to after the observation_space as Stable Baselines 3 then fails.
        # Define what the AI agent can observe.
        # Dict space gives us structured, human-readable observations.
        # 6 possible faces of the dice. Max 8 dice.
        self.observation_space = gym.spaces.Dict(
            {
                "dice_collected": gym.spaces.Box(low=0, high=NUM_DICE, shape=(6,), dtype=np.int64),
                "dice_rolled": gym.spaces.Box(low=0, high=NUM_DICE, shape=(6,), dtype=np.int64),
                # Flatten the tiles into a 16-length binary vector. Needed for SB3 compatibility.
                # Nested dicts are not supported by SB3.
                "tiles_table": gym.spaces.Box(low=0, high=1, shape=(16,), dtype=np.int8),
                "tile_players": gym.spaces.Box(low=0, high=LARGEST_TILE, shape=(len(self._players),), dtype=np.int8),
            }
        )
        # Action space is a tuple. First action: which dice to take. Second action: roll again or not.
        self.action_space = gym.spaces.MultiDiscrete([6, 2])

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Render the environment."""
        # pass

    def _create_players(self) -> None:
        names = ["Alfa", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
        self._players.append(self._you)
        for i in range(self._number_of_bots):
            self._players.append(Player(bot=True, name=names[i]))

    def _tiles_vector(self) -> ndarray[Any, dtype[Any]]:
        """Return tiles table as a flat binary vector of length 16 for indexes 21..36."""
        return np.array(
            [1 if self._table_tiles.get_table()[i] else 0 for i in range(21, 37)],
            dtype=np.int8,
        )

    def _current_obs(self) -> dict[str, object]:
        return {
            "dice_collected": np.array(self._dice.get_collected()),
            "dice_rolled": np.array(self._dice.get_rolled()),
            "tiles_table": self._tiles_vector(),
            # pylint: disable=bad-builtin
            "tile_players": np.array(list(map(lambda p: p.show(), self._players)), dtype=np.int8),
        }

    def _get_info(self) -> dict[str, object]:
        """Compute auxiliary information for debugging.

        Returns:
            dict: Additional information. Useful for debugging but not necessary for learning.
        """
        return_value = {
            "dice": self._dice,
            "terminated": self._terminated,
            "tiles_table_vec": self._tiles_vector(),
            "smallest_tile": self._table_tiles.smallest(),
            "explanation": self._explanation,
            "player_stack": self._players[0].show_all(),
            "player_score": self._players[0].end_score(),
            "bot_scores": [player.end_score() for player in self._players[1:]],
        }
        return return_value

    def _soft_reset(self) -> None:
        """Clear collected and rolled and roll again."""
        self._dice = Dice()
        self._checker = Checker(self._dice, self._players, self._table_tiles)
        self._failed_attempt = False
        self._dice.roll()

    def _remove_tile_from_player(self) -> int:
        return_value = 0
        if self._players[self._current_player_index].show():
            tile_to_return: int = self._players[
                self._current_player_index
            ].remove_tile()  # Remove the tile from the player.
            self._table_tiles.get_table()[tile_to_return] = True  # Return the tile to the table.
            worm_index = tile_to_return - SMALLEST_TILE
            return_value = -self._table_tiles.worm_values[worm_index]  # Reward is MINUS the value of the worm value.
            # If the returned tile is not the highest, turn the highest tile face down, by setting it to False.
            # Search for the highest tile to turn.
            highest = self._table_tiles.highest()
            # Turn the highest tile if there is one.
            if highest:
                self._table_tiles.set_tile(highest, False)
        return return_value

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT:Must call this first. Seed the random number generator.
        super().reset(seed=seed)
        self._dice = Dice()
        self._checker = Checker(self._dice, self._players, self._table_tiles)
        self._you = Player(bot=False, name="You")
        self._players = []
        self._create_players()
        self._table_tiles = TableTiles()
        self._failed_attempt = False
        self._terminated = False
        self._truncated = False
        self._dice.roll()
        return self._current_obs(), self._get_info()

    def _step_dice(self) -> None:
        """Execute one roll of the dice and picking or returning a tile.

        :param: action: The action to take: which dice to collect.
        """
        self._failed_attempt, self._explanation = self._checker.set_failed_already_collected()

        self._dice.collect(self._action[ACTION_INDEX_DICE])

        self._failed_attempt, self._explanation = self._checker.set_failed_no_tile_to_take(
            self._current_player_index, self._action
        )
        self._failed_attempt, self._explanation = self._checker.set_failed_no_worms(self._action)

        # Action is to roll
        if self._action[ACTION_INDEX_ROLL] == ACTION_ROLL:
            self._dice.roll()
            self._failed_attempt, self._explanation = self._checker.set_failed_already_collected()
            self._failed_attempt, self._explanation = self._checker.set_failed_no_tile_to_take(
                self._current_player_index, self._action
            )
            self._failed_attempt, self._explanation = self._checker.set_failed_no_worms(self._action)

    def _steal_from_bot(self, steal_index: int) -> int:
        tile_to_return: int = self._players[steal_index].remove_tile()  # Remove the tile from the player.
        self._players[self._current_player_index].add_tile(tile_to_return)
        worm_index = tile_to_return - SMALLEST_TILE
        return self._table_tiles.worm_values[worm_index]

    def _step_tiles(self) -> int:
        """Pick or return a tile.

        Internal sub-step for picking or returning a tile after finishing rolling dice.

        :return: Value of moving the tile [-4 to +4]
        """
        dice_sum: int = self._dice.score()[0]
        # Using the dice sum as an index in [21..36] below. Hence, for dice_sum < 21 need to return early.
        # A failed attempt or 21 was not reached. Return the tile to the table.
        if self._failed_attempt:
            return_value = self._remove_tile_from_player()
            self._soft_reset()
            return return_value
        # Environment takes the highest tile on the table.
        # Check if any tile can be picked from another player.
        # Index from player to steal.
        steal_index = next(
            (
                i
                for i, player in enumerate(self._players)
                if i != self._current_player_index and player.show() == dice_sum
            ),
            None,
        )
        if steal_index is not None:
            return_value = self._steal_from_bot(steal_index)

        # Only pick a tile if it is on the table.
        elif self._table_tiles.get_table()[dice_sum]:
            self._players[self._current_player_index].add_tile(dice_sum)  # Add the tile to the player or bot.
            self._table_tiles.set_tile(dice_sum, False)  # Mark the tile as no longer on the table.
            worm_index = dice_sum - SMALLEST_TILE
            return_value = self._table_tiles.worm_values[worm_index]
        # Tile is not available on the table
        else:
            # Pick the highest of the tiles smaller than the unavailable tile
            # Find the highest tile smaller than the dice sum.
            highest: int = self._table_tiles.find_next_lower_tile(dice_sum)
            if highest:  # Found the highest tile to pick from the table.
                self._players[self._current_player_index].add_tile(highest)  # Add the tile to the player.
                self._table_tiles.set_tile(highest, False)  # Mark the tile as no longer on the table.
                worm_index = highest - SMALLEST_TILE
                return_value = self._table_tiles.worm_values[worm_index]
            # No smaller tiles are available -> have to return players showing tile if there is one.
            else:
                return_value = self._remove_tile_from_player()
                self._explanation = RED + "No available tile on the table to take" + NO_RED

        self._soft_reset()
        return return_value

    def _play_bot(self) -> None:
        """Play a bot if there is one."""
        bot = Bot()
        bot_action: tuple[int, int] = 0, 0
        for player in self._players[1:]:
            if player.bot:
                # pylint: disable=while-used
                while bot_action[1] == 0 and not self._terminated and not self._failed_attempt:
                    bot_action = bot.policy(
                        self._dice.get_rolled(),
                        self._dice.get_collected(),
                        self._table_tiles.smallest(),
                    )
                    self._step_bot(bot_action)
            bot_action = 0, 0
            self._current_player_index += 1

    def _step_bot(self, action: tuple[int, int]) -> None:
        """Step the bot."""
        self._action = action
        self._terminated, self._truncated, self._explanation = self._checker.action_is_allowed(action)

        # Stop immediately if action was not allowed or similar.
        if self._terminated or self._truncated:
            return

        # Collect and roll the dice.
        self._step_dice()

        # Stopp rolling, move tile.
        if self._action[ACTION_INDEX_ROLL] == ACTION_STOP or self._failed_attempt:
            self._step_tiles()
            self._soft_reset()

        # Game over check.
        if not self._table_tiles.highest():
            self._terminated = True
            self._explanation = f"{GREEN}No Tile on the table, game over.{NO_GREEN}"

    def step(self, action: tuple[int, int]) -> tuple[dict[str, Any], int, bool, bool, dict[str, object]]:
        """Take a step in the environment."""
        self._action = action
        reward = 0
        # Check legal move before doing a step.
        self._terminated, self._truncated, self._explanation = self._checker.action_is_allowed(action)

        # Game Over if no Tile is on the table anymore.
        if not self._table_tiles.highest():
            self._terminated = True
            self._explanation = GREEN + "No Tile on the table, GAME OVER!" + NO_GREEN

        if self._terminated or self._truncated:
            return (
                self._current_obs(),
                0,
                self._terminated,
                self._truncated,
                self._get_info(),
            )

        # Collect and roll the dice
        self._step_dice()

        # Action is to stop or failed attempt, get reward from step tiles.
        if self._action[ACTION_INDEX_ROLL] == ACTION_STOP or self._failed_attempt:
            # self._set_failed_attempt()
            reward = self._step_tiles()
            self._current_player_index = 1
            self._play_bot()
            self._current_player_index = 0

        return (
            self._current_obs(),
            reward,
            self._terminated,
            self._truncated,
            self._get_info(),
        )


if __name__ == "__main__":
    print("This is the pickomino environment file. It is not intended to be used directly. To play run main.py.")
    env = PickominoEnv(1)
    game_observation, game_info = env.reset()
    print(game_observation, game_info)
