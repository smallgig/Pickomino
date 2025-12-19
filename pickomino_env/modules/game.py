"""Class Game."""

# ruff: noqa: I001

from __future__ import annotations

from typing import TYPE_CHECKING

from pickomino_env.modules.dice import Dice
from pickomino_env.modules.tiles import Tiles
from pickomino_env.modules.constants import (
    ACTION_INDEX_DICE,
    ACTION_INDEX_ROLL,
    ACTION_ROLL,
    ACTION_STOP,
    GREEN,
    NO_GREEN,
    NO_RED,
    NUM_DICE,
    RED,
    SMALLEST_TILE,
)

if TYPE_CHECKING:
    import numpy as np

__all__ = ["Game"]


class Game:  # pylint: disable=too-few-public-methods, disable=too-many-instance-attributes.
    """Class Game."""

    class Player:
        """Player class with his tiles and name."""

        def __init__(self, *, bot: bool, name: str) -> None:
            """Initialize a player."""
            self.name: str = name
            self.tile_stack: list[int] = []
            self.bot: bool = bot

        def show(self) -> int:
            """Show the tile from the player stack."""
            if self.tile_stack:
                return self.tile_stack[-1]
            return 0

        def show_all(self) -> list[int]:
            """Show all tiles on the player stack."""
            if self.tile_stack:
                return self.tile_stack
            return [0]  # Zero indicates the stack is empty.

        def add_tile(self, tile: int) -> None:
            """Add a tile to the player stack."""
            self.tile_stack.append(tile)

        def remove_tile(self) -> int:
            """Remove the top tile from the player stack."""
            return self.tile_stack.pop()

        def end_score(self) -> int:
            """Return player score at the end of the game."""
            score: int = 0
            tiles = Tiles()
            for tile in self.tile_stack:
                score += tiles.worm_values[tile - SMALLEST_TILE]  # List of worm values count from zero.
            return score

    class RuleChecker:
        """Class RuleChecker."""

        def __init__(self, dice: Dice, players: list[Game.Player], table_tiles: Tiles) -> None:
            """Initialize RuleChecker."""
            self._failed_attempt = False
            self._terminated = False
            self._truncated = False
            self._explanation = ""
            self._dice = dice
            self._players = players
            self._table_tiles = table_tiles

        def set_failed_already_collected(self) -> tuple[bool, str]:
            """Check if a die is available to take."""
            can_take = any(
                rolled > 0 and collected == 0
                for rolled, collected in zip(
                    self._dice.get_rolled(),
                    self._dice.get_collected(),
                )
            )

            self._failed_attempt = not can_take
            self._explanation = (
                GREEN + "Good case" + NO_GREEN
                if can_take
                else RED + f"Failed: Collected was {self._dice.get_collected()}\n"
                f"No possible rolled dice to taken in {self._dice.get_rolled()}" + NO_RED
            )

            return self._failed_attempt, self._explanation

        def set_failed_no_tile_to_take(
            self,
            current_player_index: int,
            action: tuple[int, int],
        ) -> tuple[bool, str]:
            """Failed: Not able to take a tile with the dice sum reached."""
            # Environment takes the highest tile on the table or player stack.
            # Check if any tile can be picked from another player.
            # Index from player to steal.

            steal_index = next(
                (
                    i
                    for i, player in enumerate(self._players)
                    if i != current_player_index and player.show() == self._dice.score()[0]
                ),
                None,
            )
            # pylint: disable=confusing-consecutive-elif
            if self._dice.score()[0] < SMALLEST_TILE:
                if action[ACTION_INDEX_ROLL] == ACTION_STOP:
                    self._failed_attempt = True
                    self._explanation = RED + "Failed: 21 not reached and action stop" + NO_RED

                if sum(self._dice.get_collected()) == NUM_DICE:
                    self._failed_attempt = True
                    self._explanation = RED + "Failed: 21 not reached and no dice left" + NO_RED

            # Check if no tile available on the table or from a player to take.
            elif (
                not self._table_tiles.get_tiles()[self._dice.score()[0]]
                and not self._table_tiles.find_next_lower(self._dice.score()[0])
                and steal_index is None
            ):
                self._failed_attempt = True
                self._explanation = RED + "Failed: No tile on table or from another player can be taken" + NO_RED

            return self._failed_attempt, self._explanation

        def set_failed_no_worms(self, action: tuple[int, int]) -> tuple[bool, str]:
            """Set failed attempt for no worm collected.

            No worm collected, but the action is to stop.
            """
            if not self._dice.score()[1] and action[ACTION_INDEX_ROLL] == ACTION_STOP:
                self._failed_attempt = True
                self._explanation = RED + "Failed: No worm collected" + NO_RED

            return self._failed_attempt, self._explanation

    class ActionChecker:
        """Class ActionChecker."""

        def __init__(self, dice: Dice) -> None:
            """Initialize ActionChecker."""
            self._terminated = False
            self._truncated = False
            self._explanation = ""
            self._dice = dice

        def action_is_allowed(self, action: tuple[int, int]) -> tuple[bool, bool, str]:
            """Check if action is allowed."""
            self._terminated = False
            self._truncated = False

            # Check action values are within range
            if action[ACTION_INDEX_DICE] not in range(6) or action[ACTION_INDEX_ROLL] not in range(2):
                self._terminated = True
                self._explanation = RED + "Terminated: Action index not in range" + NO_RED
                return self._terminated, self._truncated, self._explanation

            # Selected Face value was not rolled.
            if self._dice.get_rolled()[action[ACTION_INDEX_DICE]] == 0:
                self._truncated = True
                self._explanation = RED + "Truncated: Selected Face value not rolled" + NO_RED
                return self._terminated, self._truncated, self._explanation

            # Dice already collected cannot be taken again.
            if self._dice.get_collected()[action[ACTION_INDEX_DICE]] != 0:
                self._truncated = True
                self._explanation = RED + "Truncated: Dice already collected cannot be taken again" + NO_RED
                return self._terminated, self._truncated, self._explanation

            remaining_dice = self._dice.get_rolled().copy()  # Copy in order not to overwrite the real rolled variable.
            remaining_dice[action[ACTION_INDEX_DICE]] = 0  # Overwrite with zero for the last face collected.

            # Try to roll when no dice left to roll.
            if action[ACTION_INDEX_ROLL] == ACTION_ROLL and not remaining_dice:
                self._truncated = True
                self._explanation = RED + "Truncated: No Dice left to roll and roll action selected." + NO_RED
                return self._terminated, self._truncated, self._explanation

            return self._terminated, self._truncated, self._explanation
            # Get to here:Action allowed try to take a tile.

    def __init__(self, random_generator: np.random.Generator | None = None) -> None:
        """Initialize Game."""
        self.dice: Dice = Dice(random_generator)
        self.table_tiles: Tiles = Tiles()
        self.you: Game.Player = Game.Player(bot=False, name="You")
        self.players: list[Game.Player] = []
        self.action_checker = Game.ActionChecker(self.dice)
        self.rule_checker = Game.RuleChecker(self.dice, self.players, self.table_tiles)
        self.terminated: bool = False
        self.truncated: bool = False
        self.failed_attempt: bool = False  # Candidate for class RuleChecker.
        self.explanation: str = "Constructor"  # The reason, why the terminated, truncated or failed attempt is set.
        self.current_player_index: int = 0  # 0 for the player, 1 or more for bots.
