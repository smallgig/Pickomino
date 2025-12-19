"""Class Game."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pickomino_env.modules.constants import (
    ACTION_INDEX_DICE,
    ACTION_INDEX_ROLL,
    ACTION_ROLL,
    NO_RED,
    RED,
)
from pickomino_env.modules.dice import Dice
from pickomino_env.modules.player import Player
from pickomino_env.modules.rule_checker import RuleChecker
from pickomino_env.modules.tiles import Tiles

if TYPE_CHECKING:
    import numpy as np

__all__ = ["Game"]


class Game:  # pylint: disable=too-few-public-methods, disable=too-many-instance-attributes.
    """Class Game."""

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
        self.tiles: Tiles = Tiles()
        self.you: Player = Player(bot=False, name="You")
        self.players: list[Player] = []
        self.action_checker = Game.ActionChecker(self.dice)
        self.rule_checker = RuleChecker(self.dice, self.players, self.tiles)
        self.terminated: bool = False
        self.truncated: bool = False
        self.failed_attempt: bool = False  # Candidate for class RuleChecker.
        self.explanation: str = "Constructor"  # The reason, why the terminated, truncated or failed attempt is set.
        self.current_player_index: int = 0  # 0 for the player, 1 or more for bots.
