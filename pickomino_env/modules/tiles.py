"""Class for handling tiles on the table."""

from __future__ import annotations

__all__ = ["Tiles"]


class Tiles:
    """Handel the tiles on the table."""

    def __init__(self) -> None:
        """Construct the tiles on the table."""
        self.tiles: dict[int, bool] = {
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

    def set_tile(self, *, tile_number: int, is_available: bool) -> None:
        """Set one Tile."""
        self.tiles[tile_number] = is_available

    def get_tiles(self) -> dict[int, bool]:
        """Get the tile table."""
        return self.tiles

    def is_empty(self) -> bool:
        """Check if the table is empty."""
        return not self.tiles.values()

    def highest(self) -> int:
        """Get the highest tile on the table."""
        highest = 0
        if not self.is_empty():
            for key, value in self.tiles.items():
                if value:
                    highest = key
        return highest

    def smallest(self) -> int:
        """Get the smallest tile on the table."""
        smallest = 0
        if not self.is_empty():
            for key, value in reversed(self.tiles.items()):
                if value:
                    smallest = key
        return smallest

    def find_next_lower(self, dice_sum: int) -> int:
        """Find the next lower tile than the dice sum."""
        lowest = 0
        for key, value in self.tiles.items():
            if key < dice_sum and value:
                lowest = key
        return lowest
