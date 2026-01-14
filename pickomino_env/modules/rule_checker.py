"""Provide methods to check the rules of the game Pickomino."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pickomino_env.modules.constants import (
    ACTION_INDEX_ROLL,
    ACTION_STOP,
    GREEN,
    NO_GREEN,
    NO_RED,
    NUM_DICE,
    RED,
    SMALLEST_TILE,
)

if TYPE_CHECKING:
    from pickomino_env.modules.dice import Dice
    from pickomino_env.modules.player import Player
    from pickomino_env.modules.tiles import Tiles

__all__ = ["RuleChecker"]


class RuleChecker:
    """Class RuleChecker."""

    def __init__(self, dice: Dice, players: list[Player], table_tiles: Tiles) -> None:
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

        if can_take:
            self._explanation = GREEN + "Good case" + NO_GREEN
        else:
            self._explanation = (
                RED + f"Failed: Collected was {self._dice.get_collected()}\n"
                f"No possible rolled dice to taken in {self._dice.get_rolled()}" + NO_RED
            )

        return self._failed_attempt, self._explanation

    def set_failed_no_tile_to_take(self, current_player_index: int, action: tuple[int, int]) -> tuple[bool, str]:
        """Failed: Not able to take a tile with the dice sum reached."""
        if self._check_below_minimum_score(action):
            return self._failed_attempt, self._explanation

        if self._check_no_tile_available(current_player_index):
            return self._failed_attempt, self._explanation

        return self._failed_attempt, self._explanation

    def _check_below_minimum_score(self, action: tuple[int, int]) -> None:
        """Check if score is below minimum."""
        if self._dice.score()[0] < SMALLEST_TILE:
            if action[ACTION_INDEX_ROLL] == ACTION_STOP:
                self._failed_attempt = True
                self._explanation = RED + "Failed: 21 not reached and action stop" + NO_RED
            if sum(self._dice.get_collected()) == NUM_DICE:
                self._failed_attempt = True
                self._explanation = RED + "Failed: 21 not reached and no dice left" + NO_RED

    def _check_no_tile_available(self, current_player_index: int) -> None:
        """Check if no tile can be taken."""
        if self._dice.score()[0] < SMALLEST_TILE:
            return None

        steal_index = next(
            (
                i
                for i, player in enumerate(self._players)
                if i != current_player_index and player.show() == self._dice.score()[0]
            ),
            None,
        )
        # Check if no tile available on the table or from a player to take.
        if (
            not self._table_tiles.get_tiles()[self._dice.score()[0]]
            and not self._table_tiles.find_next_lower(self._dice.score()[0])
            and steal_index is None
        ):
            self._failed_attempt = True
            self._explanation = RED + "Failed: No tile on table or from another player can be taken" + NO_RED


    def set_failed_no_worms(self, action: tuple[int, int]) -> tuple[bool, str]:
        """Set failed attempt for no worm collected.

        No worm collected, but the action is to stop.
        """
        if not self._dice.score()[1] and action[ACTION_INDEX_ROLL] == ACTION_STOP:
            self._failed_attempt = True
            self._explanation = RED + "Failed: No worm collected" + NO_RED

        return self._failed_attempt, self._explanation
