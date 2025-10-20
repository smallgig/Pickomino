"""Functions used in different files."""

SMALLEST_TILE = 21
LARGEST_TILE = 36


def get_worms(moved_key: int) -> int:
    """Give back the number of worms, 1..4, for given the dice sum, 21..36.

    Mapping:
    21–24 -> 1, 25–28 -> 2, 29–32 -> 3, 33–36 -> 4.
    """
    if not SMALLEST_TILE <= moved_key <= LARGEST_TILE:
        raise ValueError("dice_sum must be between 21 and 36.")
    return (moved_key - SMALLEST_TILE) // 4 + 1


if __name__ == "__main__":
    print("This is the utils file.")
    print(get_worms(21))  # Using the get_worms function to avoid pylint messages.
