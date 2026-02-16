"""Bot class."""

from __future__ import annotations

__all__ = ["Bot"]

from typing import TYPE_CHECKING

import numpy as np

from pickomino_env.modules.constants import (
    ACTION_ROLL,
    ACTION_STOP,
    MIN_ROLLS_FOR_WORM_STRATEGY,
    WORM_INDEX,
    WORM_VALUE,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# pylint: disable=too-few-public-methods
class Bot:
    """Bot class."""

    HEURISTIC = "heuristic"

    def __init__(self) -> None:
        """Initialize Bots."""
        self.roll_counter: int = 0
        self.current_policy: str = self.HEURISTIC

    def policy(
        self,
        rolled: list[int],
        collected: list[int],
        smallest: int,
    ) -> tuple[int, int]:
        """Policy function."""
        action = ACTION_ROLL, WORM_INDEX
        if self.current_policy == self.HEURISTIC:
            action = self._heuristic_policy(rolled, collected, smallest)
        return action

    def _heuristic_policy(
        self,
        rolled: list[int],
        collected: list[int],
        smallest: int,
    ) -> tuple[int, int]:
        """Heuristic Strategy.

        1. On or after the third roll, take worms if you can.
        2. Otherwise, take the die side that contributes the most points.
        3. Quit as soon as you can take a tile.
        """
        action_roll: int = ACTION_ROLL
        self.roll_counter += 1
        values: list[int] = [1, 2, 3, 4, 5, WORM_VALUE]
        rolled = np.array(rolled)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

        if sum(collected):
            self.roll_counter = 0

        # Set rolled[index] to 0 if already collected
        for index, die in enumerate(collected):
            if die:
                rolled[index] = 0
        # 2. Otherwise, take the die side that contributes the most points.
        contribution: NDArray[np.int_] = np.multiply(rolled, values)
        max_value: int = int(np.max(contribution))  # pyright:ignore[reportUnknownMemberType, reportUnknownArgumentType]
        # All faces with max contribution.
        candidates: NDArray[np.int_] = np.where(contribution == max_value)[0]

        # Prefer worms over 5's by reversing the array.
        candidates = np.flip(candidates)

        # If they are equal, take the face with the lowest dice.
        action_dice = int(candidates[np.argmin(rolled[candidates])])

        # 1. On or after the third roll, take worms if you can.
        if self.roll_counter >= MIN_ROLLS_FOR_WORM_STRATEGY and not collected[WORM_INDEX] and rolled[WORM_INDEX]:
            action_dice = WORM_INDEX

        # 3. Quit as soon as you can take a tile: dice sum for a visible tile reached and worm collected.
        worm_collected = bool(collected[WORM_INDEX]) or action_dice == WORM_INDEX
        if sum(np.multiply(collected, values)) + contribution[action_dice] >= smallest and worm_collected:
            action_roll = ACTION_STOP

        return action_dice, action_roll
