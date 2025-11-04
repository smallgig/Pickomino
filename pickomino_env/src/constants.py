"""Project wide constants."""

from typing import Final

# Coloured printouts.
RED: Final[str] = "\033[31m"
NO_RED: Final[str] = "\033[0m"
GREEN: Final[str] = "\033[32m"
NO_GREEN: Final[str] = "\033[0m"

# Game constants.
SMALLEST_TILE: Final[int] = 21
LARGEST_TILE: Final[int] = 36
NUM_DICE: Final[int] = 8

# Action constants.
ACTION_INDEX_DICE: Final[int] = 0
ACTION_INDEX_ROLL: Final[int] = 1
ACTION_ROLL: Final[int] = 0
ACTION_STOP: Final[int] = 1


if __name__ == "__main__":
    print("Named constants:")
    print("RED:", RED)
    print("NO_RED:", NO_RED)
    print("GREEN:", GREEN)
    print("NO_GREEN:", NO_GREEN)
    print("SMALLEST_TILE:", SMALLEST_TILE)
    print("LARGEST_TILE:", LARGEST_TILE)
    print("NUM_DICE:", NUM_DICE)
    print("ACTION_INDEX_DICE:", ACTION_INDEX_DICE)
    print("ACTION_INDEX_ROLL:", ACTION_INDEX_ROLL)
    print("ACTION_ROLL:", ACTION_ROLL)
    print("ACTION_STOP:", ACTION_STOP)
    print("\n")
