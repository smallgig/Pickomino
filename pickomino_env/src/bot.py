"""Bot class."""

import numpy as np
from numpy.ma.core import argmax


class Bot:
    """Bot class."""

    HEURISTIC = "heuristic"

    def __init__(self) -> None:
        self.roll_counter: int = 0
        self.current_policy: str = self.HEURISTIC

    def __str__(self) -> str:
        """Return a string representation."""
        return f"Bot policy: {self.current_policy}"

    def set_policy(self, policy: str) -> None:
        """Set the bot policy."""
        self.current_policy = policy

    def get_policy(self) -> str:
        """Get the bot policy."""
        return self.current_policy

    def policy(self, rolled: list[int], collected: list[int], smallest: int) -> tuple[int, int]:
        """Policy function."""
        action = 0, 0
        if self.current_policy == self.HEURISTIC:
            action = self._heuristic_policy(rolled, collected, smallest)
        return action

    def _heuristic_policy(self, rolled: list[int], collected: list[int], smallest: int) -> tuple[int, int]:
        """Heuristic Strategy.

        1. On or after the third roll, take worms if you can.
        2. Otherwise, take the die side that contributes the most points.
        3. Quit as soon as you can take a tile.
        """
        action_roll = 0
        self.roll_counter += 1
        values = [1, 2, 3, 4, 5, 5]

        if sum(collected):
            self.roll_counter = 0

        # Set rolled[ind] to 0 if already collected
        for ind, die in enumerate(collected):
            if die:
                rolled[ind] = 0
        # 2. Otherwise, take the die side that contributes the most points.
        contribution = np.multiply(rolled, values)
        action_dice = int(argmax(contribution))

        # 1. On or after the third roll, take worms if you can.
        if self.roll_counter >= 3 and not collected[5] and rolled[5]:  # pylint: disable=magic-value-comparison
            action_dice = 5

        # 3. Quit as soon as you can take a tile.
        if sum(np.multiply(collected, values)) + contribution[action_dice] >= smallest:
            action_roll = 1

        return action_dice, action_roll


def print_roll(observation: tuple[list[int], list[int]], total: int, dice: object) -> None:
    """Print one roll."""
    print(dice)
    # Print line of collected dice.
    for collected in range(len(observation[0])):
        print(f"   {observation[0][collected]}      ", end="")
    # Print sum at the end of the collected dice line
    print(f" collected   sum = {total}")
    # Print line of rolled dice.
    for rolled in range(len(observation[1])):
        print(f"   {observation[1][rolled]}      ", end="")
    print(" rolled")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    print("This is the bot file.")
    bot = Bot()  # Using the Bot class to avoid pylint messages.
    print("bot", bot)
    print(
        "bot.policy([1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0], 1)",
        bot.policy([1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0], 1),
    )
    print("bot.get_policy()", bot.get_policy())
    bot.set_policy("Test")
    print("bot.set_policy('Test')")
    print("bot.get_policy()", bot.get_policy())
    print("bot", bot)
