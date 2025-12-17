"""Pickomino game with gymnasium API."""

from __future__ import annotations

__all__ = ["PickominoEnv"]

import time
from typing import Any

import gymnasium as gym
import numpy as np

from pickomino_env.modules.bot import Bot
from pickomino_env.modules.constants import (  # Coloured printouts, game and action constants.
    ACTION_INDEX_DICE,
    ACTION_INDEX_ROLL,
    ACTION_ROLL,
    ACTION_STOP,
    LARGEST_TILE,
    NUM_DICE,
    RENDER_DELAY,
    SMALLEST_TILE,
)
from pickomino_env.modules.game import Game
from pickomino_env.modules.renderer import Renderer


class PickominoEnv(gym.Env):  # type: ignore[type-arg]
    """The environment class with Gymnasium API."""

    def __init__(self, number_of_bots: int, render_mode: str | None = None) -> None:
        """Construct the environment."""
        # The following is an idea for refactoring.
        # Have only on complex variable with the return value of the step function.
        self._action: tuple[int, int] = 0, 0  # Candidate for class RuleChecker.
        self._number_of_bots: int = number_of_bots  # Remove this and use len(self._players)-1 instead.
        self._game: Game = Game()
        self._create_players()  # Do not move this to after the observation space as Stable Baselines 3 then fails.
        # Define what the AI Agent can observe.
        # Dict space gives us structured, human-readable observations.
        # 6 possible faces of the dice. Max 8 dice.
        self.observation_space = gym.spaces.Dict(
            {
                "dice_collected": gym.spaces.Box(
                    low=0,
                    high=NUM_DICE,
                    shape=(6,),
                    dtype=np.int64,
                ),
                "dice_rolled": gym.spaces.Box(
                    low=0,
                    high=NUM_DICE,
                    shape=(6,),
                    dtype=np.int64,
                ),
                # Flatten the tiles into a 16-length binary vector. Needed for SB3 compatibility.
                # Nested dicts are not supported by SB3.
                "tiles_table": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(16,),
                    dtype=np.int8,
                ),
                "tile_players": gym.spaces.Box(
                    low=0,
                    high=LARGEST_TILE,
                    shape=(len(self._game.players),),
                    dtype=np.int8,
                ),
            },
        )
        # Action space is a tuple. First action: which dice to take. Second action: roll again or not.
        self.action_space = gym.spaces.MultiDiscrete([6, 2])
        self._render_mode = render_mode
        self._renderer = Renderer(self._render_mode)

    def render(self) -> np.ndarray | list[np.ndarray] | None:  # type: ignore[override]
        """Render the environment."""
        return self._renderer.render(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            self._game.dice,
            self._game.players,
            self._game.table_tiles,
            self._game.current_player_index,
        )

    def _create_players(self) -> None:
        names = ["Alfa", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
        self._game.players.append(self._game.you)
        for i in range(self._number_of_bots):
            self._game.players.append(Game.Player(bot=True, name=names[i]))

    def _tiles_vector(self) -> np.ndarray[Any, np.dtype[Any]]:
        """Return tiles table as a flat binary vector of length 16 for indexes 21..36."""
        return np.array(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            [1 if self._game.table_tiles.get_table()[i] else 0 for i in range(21, 37)],
            dtype=np.int8,
        )

    def _current_obs(self) -> dict[str, object]:
        return {
            "dice_collected": np.array(self._game.dice.get_collected()),  # pyright: ignore[reportUnknownMemberType]
            "dice_rolled": np.array(self._game.dice.get_rolled()),  # pyright: ignore[reportUnknownMemberType]
            "tiles_table": self._tiles_vector(),
            "tile_players": np.array(  # pyright: ignore[reportUnknownMemberType]
                [p.show() for p in self._game.players],
                dtype=np.int8,
            ),
        }

    def _get_info(self) -> dict[str, object]:
        """Compute auxiliary information for debugging.

        Returns:
            dict: Additional information. Useful for debugging but not necessary for learning.

        """
        return {
            "dice_collected": list(self._game.dice.get_collected()),
            "dice_rolled": list(self._game.dice.get_rolled()),
            "terminated": self._game.terminated,
            "tiles_table_vec": self._tiles_vector(),
            "smallest_tile": self._game.table_tiles.smallest(),
            "explanation": self._game.explanation,
            "player_stack": self._game.players[0].show_all(),
            "player_score": self._game.players[0].end_score(),
            "bot_scores": [player.end_score() for player in self._game.players[1:]],
        }

    def _end_of_turn_reset(self) -> None:
        """Clear collected and rolled and roll again."""
        self._game.dice = Game.Dice()
        self._game.rule_checker = Game.RuleChecker(self._game.dice, self._game.players, self._game.table_tiles)
        self._game.action_checker = Game.ActionChecker(self._game.dice)
        self._game.failed_attempt = False
        self._game.dice.roll()

    def _remove_tile_from_player(self) -> int:
        return_value = 0
        if self._game.players[self._game.current_player_index].show():
            tile_to_return: int = self._game.players[
                self._game.current_player_index
            ].remove_tile()  # Remove the tile from the player.
            self._game.table_tiles.get_table()[tile_to_return] = True  # Return the tile to the table.
            worm_index = tile_to_return - SMALLEST_TILE
            return_value = -self._game.table_tiles.worm_values[
                worm_index
            ]  # Reward is MINUS the value of the worm value.
            # If the returned tile is not the highest, turn the highest tile face down by setting it to False.
            # Search for the highest tile to turn.
            highest = self._game.table_tiles.highest()
            # Turn the highest tile if there is one.
            if highest:
                self._game.table_tiles.set_tile(tile_number=highest, is_available=False)
        return return_value

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: RUF100, ARG002 external API constraint.
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            observation, info for the initial state

        """
        # IMPORTANT. Must call this first. Seed the random number generator.
        super().reset(seed=seed)
        self._game = Game(random_generator=self.np_random)
        self._create_players()
        self._game.dice.roll()
        return self._current_obs(), self._get_info()

    def _step_dice(self) -> None:
        """Execute one roll of the dice and picking or returning a tile.

        :param: action: The action to take: which dice to collect.
        """
        self._game.failed_attempt, self._game.explanation = self._game.rule_checker.set_failed_already_collected()

        self._game.dice.collect(self._action[ACTION_INDEX_DICE])

        self._game.failed_attempt, self._game.explanation = self._game.rule_checker.set_failed_no_tile_to_take(
            self._game.current_player_index,
            self._action,
        )
        self._game.failed_attempt, self._game.explanation = self._game.rule_checker.set_failed_no_worms(
            self._action,
        )

        # Action is to roll
        if self._action[ACTION_INDEX_ROLL] == ACTION_ROLL:
            self._game.dice.roll()
            self._game.failed_attempt, self._game.explanation = self._game.rule_checker.set_failed_already_collected()
            self._game.failed_attempt, self._game.explanation = self._game.rule_checker.set_failed_no_tile_to_take(
                self._game.current_player_index,
                self._action,
            )
            self._game.failed_attempt, self._game.explanation = self._game.rule_checker.set_failed_no_worms(
                self._action,
            )

    def _steal_from_bot(self, steal_index: int) -> int:
        tile_to_return: int = self._game.players[steal_index].remove_tile()  # Remove the tile from the player.
        self._game.players[self._game.current_player_index].add_tile(tile_to_return)
        worm_index = tile_to_return - SMALLEST_TILE
        return self._game.table_tiles.worm_values[worm_index]

    def _step_tiles(self) -> int:
        """Pick or return a tile.

        Internal sub-step for picking or returning a tile after finishing rolling dice.

        :return: Value of moving the tile [-4 to +4]
        """
        dice_sum: int = self._game.dice.score()[0]
        # Using the dice sum as an index in [21..36] below. Hence, for dice_sum < 21 need to return early.
        # A failed attempt or 21 was not reached. Return the tile to the table.
        if self._game.failed_attempt:
            return_value = self._remove_tile_from_player()
            self._end_of_turn_reset()
            return return_value
        # Environment takes the highest tile on the table.
        # Check if any tile can be picked from another player.
        # Index from player to steal.
        steal_index = next(
            (
                i
                for i, player in enumerate(self._game.players)
                if i != self._game.current_player_index and player.show() == dice_sum
            ),
            None,
        )
        if steal_index is not None:
            return_value = self._steal_from_bot(steal_index)

        # Only pick a tile if it is on the table.
        elif self._game.table_tiles.get_table()[dice_sum]:
            self._game.players[self._game.current_player_index].add_tile(
                dice_sum,
            )  # Add the tile to the player or bot.
            self._game.table_tiles.set_tile(
                tile_number=dice_sum,
                is_available=False,
            )  # Mark the tile as no longer on the table.
            worm_index = dice_sum - SMALLEST_TILE
            return_value = self._game.table_tiles.worm_values[worm_index]
        # Tile is not available on the table
        else:
            # Pick the highest of the tiles smaller than the unavailable tile
            # Find the highest tile smaller than the dice sum.
            highest: int = self._game.table_tiles.find_next_lower_tile(dice_sum)
            if highest:  # Found the highest tile to pick from the table.
                self._game.players[self._game.current_player_index].add_tile(
                    highest,
                )  # Add the tile to the player.
                self._game.table_tiles.set_tile(
                    tile_number=highest,
                    is_available=False,
                )  # Mark the tile as no longer on the table.
                worm_index = highest - SMALLEST_TILE
                return_value = self._game.table_tiles.worm_values[worm_index]
            # No smaller tiles are available -> have to return players top tile if there is one.
            else:
                return_value = self._remove_tile_from_player()
                self._game.explanation = "No available tile on the table to take"

        self._end_of_turn_reset()
        return return_value

    def _play_bot(self) -> None:
        """Play a bot if there is one."""
        bot = Bot()
        bot_action: tuple[int, int] = 0, 0
        for player in self._game.players[1:]:
            if player.bot:
                # pylint: disable=while-used
                while bot_action[1] == 0 and not self._game.terminated and not self._game.failed_attempt:
                    bot_action = bot.policy(
                        self._game.dice.get_rolled(),
                        self._game.dice.get_collected(),
                        self._game.table_tiles.smallest(),
                    )
                    self._step_bot(bot_action)
                if self._render_mode is not None:
                    self.render()  # pyright: ignore[reportUnknownMemberType]
                    time.sleep(RENDER_DELAY)
            bot_action = 0, 0
            self._game.current_player_index += 1

    def _step_bot(self, action: tuple[int, int]) -> None:
        """Step the bot."""
        self._action = action
        self._game.terminated, self._game.truncated, self._game.explanation = (
            self._game.action_checker.action_is_allowed(action)
        )

        # Stop immediately if action was not allowed or similar.
        if self._game.terminated or self._game.truncated:
            self._end_of_turn_reset()
            return

        # Collect and roll the dice.
        self._step_dice()

        # Stopp rolling, move tile.
        if self._action[ACTION_INDEX_ROLL] == ACTION_STOP or self._game.failed_attempt:
            self._step_tiles()
            self._end_of_turn_reset()

        # Game over check.
        if not self._game.table_tiles.highest():
            self._game.terminated = True
            self._game.explanation = "No Tile on the table, game over."

    def step(
        self,
        action: tuple[int, int],
    ) -> tuple[dict[str, Any], int, bool, bool, dict[str, object]]:
        """Take a step in the environment."""
        self._action = action
        reward = 0
        # Check legal move before doing a step.
        self._game.terminated, self._game.truncated, self._game.explanation = (
            self._game.action_checker.action_is_allowed(action)
        )

        # Illegal move
        if self._game.terminated or self._game.truncated:
            return (
                self._current_obs(),
                0,
                self._game.terminated,
                self._game.truncated,
                self._get_info(),
            )

        # Collect and roll the dice
        self._step_dice()

        # Action is to stop or failed attempt, get reward from step tiles.
        if self._action[ACTION_INDEX_ROLL] == ACTION_STOP or self._game.failed_attempt:
            reward = self._step_tiles()
            self._game.current_player_index = 1
            self._play_bot()
            self._game.current_player_index = 0

        # Game Over if no Tile is on the table anymore.
        if not self._game.table_tiles.highest():
            self._game.terminated = True
            self._game.explanation = "No Tile on the table, GAME OVER!"

        return (
            self._current_obs(),
            reward,
            self._game.terminated,
            self._game.truncated,
            self._get_info(),
        )

    def close(self) -> None:
        """Close the environment and renderer."""
        self._renderer.close()
