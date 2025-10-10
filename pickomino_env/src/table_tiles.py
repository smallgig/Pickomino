"""Class table_tiles"""

from mypy.modulefinder import highest_init_level
from networkx.algorithms.lowest_common_ancestors import lowest_common_ancestor


class TableTiles:
    """Define the Tiles on the Table"""

    def __init__(self) -> None:
        """Constructor"""
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

    def set_tile(self, tile_number: int, truth_value: bool) -> None:
        """Set one Tile."""
        self._tile_table[tile_number] = truth_value

    def get_table(self) -> dict[int, bool]:
        """Get whole table."""
        return self._tile_table

    def is_empty(self) -> bool:
        """Check if the table is empty"""
        if self._tile_table.values():
            return False
        return True

    def highest(self) -> int:
        """Get the highest tile on table"""
        highest = 0
        if not self.is_empty():
            for key, value in self._tile_table.items():
                if value:
                    highest = key
        return highest

    def smallest(self):
        """Get the smallest tile on table"""
        smallest = 0
        if not self.is_empty():
            for key, value in reversed(self._tile_table.items()):
                if value:
                    smallest = key
        return smallest

    def find_next_lower_tile(self, dice_sum) -> int:
        highest = 0
        for key, value in self._tile_table.items():
            if key < dice_sum and value:
                highest = key

        return highest
