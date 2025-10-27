"""Class player."""

from pickomino_env.src.table_tiles import TableTiles


class Player:
    """Player class with his tiles and name."""

    def __init__(self, bot: bool, name: str) -> None:
        """Initialize a player."""
        self.name: str = name
        self.tile_stack: list[int] = []
        self.bot: bool = bot

    def show(self) -> int:
        """Show the tile from the player stack."""
        if self.tile_stack:
            return self.tile_stack[-1]
        return 0

    def show_all(self) -> list[int]:
        """Show all tiles on the player stack."""
        if self.tile_stack:
            return self.tile_stack
        return [42]

    def add_tile(self, tile: int) -> None:
        """Add a tile to the player stack."""
        self.tile_stack.append(tile)

    def remove_tile(self) -> int:
        """Remove the top tile from the player stack."""
        return self.tile_stack.pop()

    def score(self) -> int:
        """Return player score at the end of the game."""
        score: int = 0
        table = TableTiles()
        for tile in self.tile_stack:
            score += table.worm_values[tile - 21]  # List of worm values count from zero.
        return score


if __name__ == "__main__":
    print("This is the player file.")
    player = Player(True, "Dummy")  # Using the Player class to avoid pylint messages.
    print(player.show())
