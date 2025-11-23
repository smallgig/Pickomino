"""Renderer using pygame."""

__all__ = ["Renderer"]
# pygame internally uses deprecated pkg_resources
# See: https://setuptools.pypa.io/en/latest/pkg_resources.html
import warnings
from importlib.resources import files

import numpy as np

from pickomino_env.modules.constants import (
    LARGEST_TILE,
    RENDER_MODE_HUMAN,
    RENDER_MODE_RGB_ARRAY,
    SMALLEST_TILE,
)
from pickomino_env.modules.dice import Dice
from pickomino_env.modules.player import Player
from pickomino_env.modules.table_tiles import TableTiles

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
# E402: module level import not at the top of the file. Needed to suppress warning before import.
import pygame  # noqa: E402 # pylint: disable=wrong-import-position, wrong-import-order


class Renderer:  # pylint: disable=too-many-instance-attributes
    """Class Renderer."""

    def __init__(self, render_mode: str | None = None) -> None:
        self._render_mode: str | None = render_mode
        self._window: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None

        # Screen size
        self._size: tuple[int, int] = (800, 600)  # (width, height)

        self._dice: Dice | None = None
        self._players: list[Player] | None = None
        self._tiles: TableTiles | None = None
        self._sprite_dir = files("pickomino_env").joinpath("sprites")

    def render(self, dice: Dice, players: list[Player], tiles: TableTiles) -> np.ndarray | list[np.ndarray] | None:
        """Render the environment."""
        self._dice = dice
        self._players = players
        self._tiles = tiles

        if self._render_mode is None:
            return None

        if self._render_mode == RENDER_MODE_HUMAN:
            self._render_human()
        elif self._render_mode == RENDER_MODE_RGB_ARRAY:
            return self._render_rgb_array()
        return None

    def _render_human(self) -> None:
        """Display to screen."""
        if self._window is None:
            pygame.init()
            self._window = pygame.display.set_mode(self._size)
            self._clock = pygame.time.Clock()

        # Draw background.
        # self._window.fill((34, 139, 34))  # Dark green instead of white.
        self._window.fill((70, 130, 70))  # Lighter, softer green.
        # self._window.fill((100, 140, 100))  # Muted sage green.

        # Draw game state.
        self._draw_board()

        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(60)  # 60 FPS

    def _render_rgb_array(self) -> np.ndarray:
        """Return pixel array for recording."""
        # Like _render_human. But capture as an array.
        if self._window is None:
            raise RuntimeError("Window not initialised.")
        surface = pygame.surfarray.array3d(self._window)
        return np.transpose(surface, (1, 0, 2))

    def _draw_board(self) -> None:
        """Draw the game board with tiles and dice."""
        if self._window is None or self._tiles is None:
            return
        # Draw table tiles (21-36 in a grid at bottom)
        tiles = self._tiles.get_table()
        start_x, start_y = 50, 250  # Was 500, lower number = higher on screen
        tile_width, tile_height = 80, 160
        col = 0

        for tile_num in range(SMALLEST_TILE, LARGEST_TILE + 1):
            if tiles[tile_num]:  # Only draw available tiles
                x = start_x + (col % 8) * (tile_width + 10)
                y = start_y + (col // 8) * (tile_height + 10)

                tile_path = self._sprite_dir.joinpath(f"tile_{tile_num}.png")
                tile_image = pygame.image.load(str(tile_path))
                self._window.blit(tile_image, (x, y))
            col += 1

    def close(self) -> None:
        """Close game."""
        if self._window is not None:
            pygame.quit()
