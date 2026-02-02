"""Class Dice."""

from __future__ import annotations

import numpy as np

from pickomino_env.modules.constants import (
    LARGEST_TILE,
    NUM_DICE,
    NUM_DIE_FACES,
    WORM_INDEX,
    WORM_VALUE,
)

__all__ = ["Dice"]


class Dice:
    """Class Dice. Represents a collection of die face frequencies.

    An example for eight dice with six faces is: [0, 0, 3, 4, 0, 1]
    This example means that three threes, four fours and one worm die face have been collected.
    """

    def __init__(self, random_generator: np.random.Generator | None = None) -> None:
        """Initialize Dice."""
        self._random_generator = random_generator or np.random.default_rng()
        self.values: list[int] = [1, 2, 3, 4, 5, WORM_VALUE]  # Worm has value 5.
        self._n_dice: int = NUM_DICE
        self._collected: list[int] = [
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # Collected dice, up to 8 per side.
        self._rolled: list[int] = [0, 0, 0, 0, 0, 0]  # Last roll.

    def collect(self, action_face: int) -> list[int]:
        """Insert the chosen face to the collected list."""
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
            i: int = int(self._random_generator.integers(0, NUM_DIE_FACES))  # pyright:ignore[reportUnknownMemberType]
            self._rolled[i] += 1

        return self._rolled

    def score(self) -> tuple[int, bool]:
        """Count the score based on an array of die face frequencies."""
        # Check if there is at least one worm
        has_worms = self._collected[WORM_INDEX] > 0
        # Multiply the frequency of each die face with its value
        current_score = int(np.dot(self.values, self._collected))  # pyright: ignore[reportUnknownMemberType]
        # Using the dice sum as an index in [21..36], higher rolls can only pick 36 or lower.
        current_score = int(min(current_score, LARGEST_TILE))
        return current_score, has_worms
