"""Pickomino game with gymnasium API."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from numpy import dtype, ndarray

from pickomino_env.src.bot import Bot
from pickomino_env.src.dice import Dice
from pickomino_env.src.player import Player
from pickomino_env.src.table_tiles import TableTiles

RED = "\033[31m"
NO_RED = "\033[0m"
GREEN = "\033[32m"
NO_GREEN = "\033[0m"


class PickominoEnv(gym.Env):  # type: ignore[type-arg] # pylint: disable=too-many-instance-attributes.
    """The environment class."""

    SMALLEST_TILE = 21
    LARGEST_TILE = 36
    ACTION_INDEX_DICE = 0
    ACTION_INDEX_ROLL = 1
    ACTION_ROLL = 0
    ACTION_STOP = 1
    NUM_DICE = 8

    def __init__(self, number_of_bots: int) -> None:
        """Construct the environment."""
        # The following is an idea for refactoring.
        # Have only on complex variable with the return value of the step function.
        self._action: tuple[int, int] = 0, 0  # Candidate for class Checker.
        self._roll_counter: int = 0  # This is not used.
        self._number_of_bots: int = number_of_bots  # Remove this and use len(self._players) - 1 instead.
        self._you: Player = Player(bot=False, name="You")  # Put this in the players list and remove it from here.
        self._players: list[Player] = []
        self._create_players()  # Put this function call after the variable initializations.
        self._remaining_dice: int = self.NUM_DICE  # Get rid of this. We do not need it.
        self._terminated: bool = False
        self._truncated: bool = False
        self._failed_attempt: bool = False  # Candidate for class Checker.
        self._explanation: str = "Constructor"  # Why the terminated, truncated or failed attempt is set.
        self._current_player_index: int = 0  # 0 for the player, 1 or more for bots.
        self._last_returned_tile: int = 0  # For info.
        self._last_picked_tile: int = 0  # For info.
        self._last_turned_tile: int = 0  # For infor.
        self._dice: Dice = Dice()
        self._table_tiles: TableTiles = (
            TableTiles()
        )  # Consider a complex class Table consisting of table tiles and players tiles.

        # Define what the AI agent can observe.
        # Dict space gives us structured, human-readable observations.
        # 6 possible faces of the dice. Max 8 dice.
        self.observation_space = gym.spaces.Dict(
            {
                "dice_collected": gym.spaces.Box(low=0, high=8, shape=(6,), dtype=np.int64),
                "dice_rolled": gym.spaces.Box(low=0, high=8, shape=(6,), dtype=np.int64),
                # Flatten the tiles into a 16-length binary vector. Needed for SB3 compatibility.
                # Nested dicts are not supported by SB3.
                "tiles_table": gym.spaces.Box(low=0, high=1, shape=(16,), dtype=np.int8),
                "tile_players": gym.spaces.Discrete(len(self._players)),
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

    def _get_obs_dice(
        self,
    ) -> tuple[list[int], list[int]]:
        """Convert internal state to observation format.

        Returns: Dices collected and dices rolled.
        """
        return self._dice.get_collected(), self._dice.get_rolled()

    def _get_obs_tiles(self) -> tuple[int, dict[int, bool]]:
        """Convert internal state to observation format.

        Returns:
            dict: Tiles distribution
        """
        return self._players[0].show(), self._table_tiles.get_table()

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
            "tile_players": list(map(lambda p: p.show(), self._players)),  # pylint: disable=bad-builtin
        }

    def _get_info(self) -> dict[str, object]:
        """Compute auxiliary information for debugging.

        Returns:
            dict: Additional information. Useful for debugging but not necessary for learning.
        """
        return_value = {
            "remaining_dice": self._remaining_dice,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "dice": self._dice,
            "terminated": self._terminated,
            "tiles_table_vec": self._tiles_vector(),
            "smallest_tile": self._table_tiles.smallest(),
            "explanation": self._explanation,
            "player_stack": self._players[0].show_all(),
            "last_returned_tile": self._last_returned_tile,
            "last_picked_tile": self._last_picked_tile,
            "last_turned_tile": self._last_turned_tile,
        }
        return return_value

    def _soft_reset(self) -> None:
        """Clear collected and rolled and roll again."""
        self._dice = Dice()
        self._failed_attempt = False
        self._roll_counter = 0
        self._dice.roll()

    def _remove_tile_from_player(self) -> int:
        return_value = 0
        if self._players[self._current_player_index].show():
            tile_to_return: int = self._players[
                self._current_player_index
            ].remove_tile()  # Remove the tile from the player.
            self._last_returned_tile = tile_to_return
            self._table_tiles.get_table()[tile_to_return] = True  # Return the tile to the table.
            worm_index = tile_to_return - self.SMALLEST_TILE
            return_value = -self._table_tiles.worm_values[worm_index]  # Reward is MINUS the value of the worm value.
            # If the returned tile is not the highest, turn the highest tile face down, by setting it to False.
            # Search for the highest tile to turn.
            highest = self._table_tiles.highest()
            # Turn the highest tile if there is one.
            if highest:
                self._table_tiles.set_tile(highest, False)
                self._last_turned_tile = highest
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
        self._you = Player(bot=False, name="You")
        self._players = []
        self._create_players()
        self._table_tiles = TableTiles()
        self._failed_attempt = False
        self._terminated = False
        self._truncated = False
        self._last_returned_tile = 0  # For info.
        self._last_picked_tile = 0  # For info.
        self._last_turned_tile = 0  # For info.
        self._dice.roll()
        return self._current_obs(), self._get_info()

    def _action_is_allowed(self) -> None:
        """Check if action is allowed."""
        self._terminated = False
        self._truncated = False
        self._failed_attempt = False

        # Check action values are within range
        if self._action[self.ACTION_INDEX_DICE] not in range(0, 6) or self._action[
            self.ACTION_INDEX_ROLL
        ] not in range(0, 2):
            self._terminated = True
            self._explanation = RED + "Terminated: Action index not in range" + NO_RED
        # Selected Face value was not rolled.
        if self._dice.get_rolled()[self._action[self.ACTION_INDEX_DICE]] == 0:
            self._truncated = True
            self._explanation = RED + "Truncated: Selected Face value not rolled" + NO_RED

        # Dice already collected cannot be taken again.
        if self._dice.get_collected()[self._action[self.ACTION_INDEX_DICE]] != 0:
            self._truncated = True
            self._explanation = RED + "Truncated: Dice already collected cannot be taken again" + NO_RED

        remaining_dice = self._dice.get_rolled().copy()
        remaining_dice[self._action[self.ACTION_INDEX_DICE]] = 0

        if self._action[self.ACTION_INDEX_ROLL] == self.ACTION_ROLL and not remaining_dice:
            self._truncated = True
            self._explanation = RED + "Truncated: No Dice left to roll and roll action selected." + NO_RED

        # Get to here:Action allowed try to take a tile.

    def _step_dice(self) -> None:
        """Execute one roll of the dice and picking or returning a tile.

        :param: action: The action to take: which dice to collect.
        """
        self._set_failed_already_collected()

        self._dice.collect(self._action[self.ACTION_INDEX_DICE])

        self._set_failed_no_tile_to_take()
        self._set_failed_no_worms()

        # Action is to roll
        if self._action[self.ACTION_INDEX_ROLL] == self.ACTION_ROLL:
            self._dice.roll()
            self._set_failed_already_collected()
            self._set_failed_no_tile_to_take()
            self._set_failed_no_worms()

    def _steal_from_bot(self, steal_index: int) -> int:
        tile_to_return: int = self._players[steal_index].remove_tile()  # Remove the tile from the player.
        self._players[self._current_player_index].add_tile(tile_to_return)
        self._last_returned_tile = tile_to_return
        worm_index = tile_to_return - self.SMALLEST_TILE
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
            self._last_picked_tile = dice_sum
            self._players[self._current_player_index].add_tile(dice_sum)  # Add the tile to the player or bot.
            self._table_tiles.set_tile(dice_sum, False)  # Mark the tile as no longer on the table.
            worm_index = dice_sum - self.SMALLEST_TILE
            return_value = self._table_tiles.worm_values[worm_index]
        # Tile is not available on the table
        else:
            # Pick the highest of the tiles smaller than the unavailable tile
            # Find the highest tile smaller than the dice sum.
            highest: int = self._table_tiles.find_next_lower_tile(dice_sum)
            if highest:  # Found the highest tile to pick from the table.
                self._last_picked_tile = highest
                self._players[self._current_player_index].add_tile(highest)  # Add the tile to the player.
                self._table_tiles.set_tile(highest, False)  # Mark the tile as no longer on the table.
                worm_index = highest - self.SMALLEST_TILE
                return_value = self._table_tiles.worm_values[worm_index]
            # No smaller tiles are available -> have to return players showing tile if there is one.
            else:
                return_value = self._remove_tile_from_player()
                self._explanation = RED + "No available tile on the table to take" + NO_RED

        self._soft_reset()
        return return_value

    def _set_failed_already_collected(self) -> None:
        """Check if a die is available to take."""
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

    def _set_failed_no_tile_to_take(self) -> None:
        """Failed: Not able to take a tile with dice sum reached."""
        # Environment takes the highest tile on the table or player stack.
        # Check if any tile can be picked from another player.
        # Index from player to steal.
        steal_index = next(
            (
                i
                for i, player in enumerate(self._players)
                if i != self._current_player_index and player.show() == self._dice.score()[0]
            ),
            None,
        )
        # pylint: disable=confusing-consecutive-elif
        if self._dice.score()[0] < self.SMALLEST_TILE:

            if self._action[self.ACTION_INDEX_ROLL] == self.ACTION_STOP:
                self._failed_attempt = True
                self._explanation = RED + "Failed: 21 not reached and action stop" + NO_RED

            if sum(self._dice.get_collected()) == self.NUM_DICE:
                self._failed_attempt = True
                self._explanation = RED + "Failed: 21 not reached and no dice left" + NO_RED

        # Check if no tile available on the table or from player to take.
        elif not self._table_tiles.find_next_lower_tile(self._dice.score()[0]) and steal_index is None:
            self._failed_attempt = True
            self._explanation = RED + "Failed: No tile on table or from another player can be taken" + NO_RED

    def _set_failed_no_worms(self) -> None:
        """No worm collected and action stop."""
        if not self._dice.score()[1] and self._action[self.ACTION_INDEX_ROLL] == self.ACTION_STOP:
            self._failed_attempt = True
            self._explanation = RED + "Failed: No worm collected" + NO_RED

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
        self._action_is_allowed()

        # Stop immediately if action was not allowed or similar.
        if self._terminated or self._truncated:
            return

        # Collect and roll the dice.
        self._step_dice()

        # Stopp rolling, move tile.
        if self._action[self.ACTION_INDEX_ROLL] == self.ACTION_STOP or self._failed_attempt:
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
        self._action_is_allowed()

        # Game Over if no Tile is on the table anymore.
        if not self._table_tiles.highest():
            self._terminated = True
            self._explanation = GREEN + "No Tile on the table, GAME OVER!" + NO_GREEN

        # Have to keep the values to return after resetting.
        if self._terminated:
            obs, reward, terminated, truncated, info = (
                self._current_obs(),
                0,
                self._terminated,
                self._truncated,
                self._get_info(),
            )
            self.reset()
            return obs, reward, terminated, truncated, info
        if self._truncated:
            return (
                self._current_obs(),
                reward,
                self._terminated,
                self._truncated,
                self._get_info(),
            )

        # Collect and roll the dice
        self._step_dice()

        # Action is to stop or failed attempt, get reward from step tiles.
        if self._action[self.ACTION_INDEX_ROLL] == self.ACTION_STOP or self._failed_attempt:
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

        return (
            self._current_obs(),
            reward,
            self._terminated,
            self._truncated,
            self._get_info(),
        )


if __name__ == "__main__":
    env = PickominoEnv(1)
    game_observation, game_info = env.reset()
    print(game_observation, game_info)
    print()
    print("==========================")
    print()
    print("TO PLAY: run bot_test.py")
    print()
    print("===========================")
