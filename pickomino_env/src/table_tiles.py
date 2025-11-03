"""Class table tiles."""

# pylint: disable=duplicate-code


class TableTiles:
    """Define the tiles on the table."""

    def __init__(self) -> None:
        """Construct the table tiles."""
        self._tile_table: dict[int, bool] = {
            21: True,
            22: True,
            23: True,
            24: True,
            25: True,
            26: True,
            27: True,
            28: True,
            29: True,
            30: True,
            31: True,
            32: True,
            33: True,
            34: True,
            35: True,
            36: True,
        }
        self.worm_values: list[int] = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]

    def set_tile(self, tile_number: int, truth_value: bool) -> None:
        """Set one Tile."""
        self._tile_table[tile_number] = truth_value

    def get_table(self) -> dict[int, bool]:
        """Get the tile table."""
        return self._tile_table

    def is_empty(self) -> bool:
        """Check if the table is empty."""
        if self._tile_table.values():
            return False
        return True

    def highest(self) -> int:
        """Get the highest tile on the table."""
        highest = 0
        if not self.is_empty():
            for key, value in self._tile_table.items():
                if value:
                    highest = key
        return highest

    def smallest(self) -> int:
        """Get the smallest tile on the table."""
        smallest = 0
        if not self.is_empty():
            for key, value in reversed(self._tile_table.items()):
                if value:
                    smallest = key
        return smallest

    def find_next_lower_tile(self, dice_sum: int) -> int:
        """Find the next lower tile than the dice sum."""
        lowest = 0
        for key, value in self._tile_table.items():
            if key < dice_sum and value:
                lowest = key
        return lowest


if __name__ == "__main__":
    print("This is the table tiles file.")
    table = TableTiles()  # Using the TableTiles class to avoid pylint messages.
    print(table.highest())
