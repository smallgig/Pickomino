"""Class dice."""

import random as rand
import numpy as np


class Dice:
    """Represents a collection of die face frequencies.
    An example for eight dice with six faces is: (0, 0, 3, 4, 0, 1)
    This example means that three threes, four fours and one worm die face have been collected."""

    LARGEST_TILE = 36

    def __init__(self) -> None:
        self.values: list[int] = [1, 2, 3, 4, 5, 5]  # Worm has the value 5.
        self._n_dice: int = 8
        self._collected: list[int] = [0, 0, 0, 0, 0, 0]  # Collected dice, up to 8 per side.
        self._rolled: list[int] = [0, 0, 0, 0, 0, 0]  # Last roll.

    def collect(self, action_face: int) -> list[int]:
        """Insert chosen face to collected list."""
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
            self._rolled[rand.randint(0, 5)] += 1

        return self._rolled

    def score(self) -> tuple[int, bool]:
        """Count the score based on an array of die face frequencies."""
        # Check if there is at least one worm
        has_worms = self._collected[-1] > 0
        # Multiply the frequency of each die face with its value
        current_score = int(np.dot(self.values, self._collected) if self._collected else 0)
        # Using dice_sum as an index in [21..36], higher rolls can only pick 36 or lower
        current_score = min(current_score, Dice.LARGEST_TILE)
        return current_score, has_worms

    def __str__(self) -> str:
        die_faces: list[str] = [
            "",  # index = 0 doesn't have a face
            "[     ]\n[  @  ]\n[     ]",  # index 1
            "[@    ]\n[     ]\n[    @]",  # index 2
            "[@    ]\n[  @  ]\n[    @]",  # index 3
            "[@   @]\n[     ]\n[@   @]",  # index 4
            "[@   @]\n[  @  ]\n[@   @]",  # index 5
            "\033[31m[oo@@@]\033[0m\n\033[31m[WORM!]\033[0m\n\033[31m[@@@@@]\033[0m",  # index 6
        ]
        # Print one dice.
        show_values = [1, 2, 3, 4, 5, 6]
        faces = [die_faces[v].splitlines() for v in show_values]
        return_value = ""
        for line in zip(*faces):
            return_value += "   ".join(line) + "\n"
        return return_value
