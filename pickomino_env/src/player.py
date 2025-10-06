"""Class player."""

from pickomino_env.src import utils


class Player:
    """Player class with his tiles and name."""

    def __init__(self, bot: bool, name: str) -> None:
        """Constructor for a player"""
        self.name: str = name
        self.tile_stack: list[int] = []
        self.bot: bool = bot

    def show(self) -> int:
        """Show the tile from the player stack."""
        if self.tile_stack:
            return self.tile_stack[-1]
        return 0

    def add_tile(self, tile: int) -> None:
        """Add tile to the player stack"""
        self.tile_stack.append(tile)

    def remove_tile(self) -> int:
        """Remove the top tile from the player stack."""
        return self.tile_stack.pop()

    def score(self) -> int:
        """Return player score at the end of game"""
        score: int = 0
        for tile in self.tile_stack:
            score += utils.get_worms(tile)
        return score
