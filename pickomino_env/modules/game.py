"""Class Game to abstract details from the environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pickomino_env.modules.action_checker import ActionChecker
from pickomino_env.modules.dice import Dice
from pickomino_env.modules.player import Player
from pickomino_env.modules.rule_checker import RuleChecker
from pickomino_env.modules.tiles import Tiles

if TYPE_CHECKING:
    import numpy as np

__all__ = ["Game"]


class Game:  # pylint: disable=too-few-public-methods, disable=too-many-instance-attributes.
    """Class Game."""

    def __init__(self, random_generator: np.random.Generator | None = None) -> None:
        """Initialize Game."""
        self.dice: Dice = Dice(random_generator)
        self.tiles: Tiles = Tiles()
        self.you: Player = Player(bot=False, name="You")
        self.players: list[Player] = []
        self.action_checker = ActionChecker(self.dice)
        self.rule_checker = RuleChecker(self.dice, self.players, self.tiles)
        self.terminated: bool = False
        self.truncated: bool = False
        self.failed_attempt: bool = False  # Candidate for moving to class RuleChecker.
        self.explanation: str = "Constructor"  # The reason, why the terminated, truncated or failed attempt is set.
        self.current_player_index: int = 0  # 0 for the player, 1 or more for bots.
