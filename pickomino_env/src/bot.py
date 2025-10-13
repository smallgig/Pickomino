"""Bot class"""

import numpy as np
from numpy.ma.core import argmax


class Bot:
    """Bot class"""

    def __init__(self) -> None:
        self.roll_counter: int = 0

    def heuristic_policy(self, rolled, collected, smallest) -> tuple[int, int]:
        """Heuristic Strategy.
        1. On or after the third roll, take worms if you can.
        2. Otherwise, take the die side that contributes the most points.
        3. Quit as soon as you can take a tile.
        """
        action_roll = 0
        self.roll_counter += 1
        rolled = np.array(rolled)
        values = np.array([1, 2, 3, 4, 5, 5], int)

        if sum(collected):
            self.roll_counter = 0

        # Set rolled[ind] to 0 if already collected
        for ind, die in enumerate(collected):
            if die:
                rolled[ind] = 0
        # 2. Otherwise, take the die side that contributes the most points.
        contribution = rolled * values
        action_dice = int(argmax(contribution))

        # 1. On or after the third roll, take worms if you can.
        if self.roll_counter >= 3 and not collected[5] and rolled[5]:
            action_dice = 5

        # 3. Quit as soon as you can take a tile.
        if sum(collected * values) + contribution[action_dice] >= smallest:
            action_roll = 1

        return action_dice, action_roll
