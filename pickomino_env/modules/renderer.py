"""Renderer using pygame."""

__all__ = ["Renderer"]
# pygame internally uses deprecated pkg_resources
# See: https://setuptools.pypa.io/en/latest/pkg_resources.html
import warnings
from importlib.resources import files

import numpy as np

from pickomino_env.modules.constants import (
    BACKGROUND_COLOR,
    LARGEST_TILE,
    RENDER_FPS,
    RENDER_MODE_HUMAN,
    RENDER_MODE_RGB_ARRAY,
    SMALLEST_TILE,
    TILE_HEIGHT,
    TILE_SPACING,
    TILE_WIDTH,
    TILES_PER_ROW,
    TILES_START_X,
    TILES_START_Y,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
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
        self._size: tuple[int, int] = (WINDOW_WIDTH, WINDOW_HEIGHT)

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
        self._window.fill(BACKGROUND_COLOR)  # Lighter, softer green.

        # Draw game state.
        self._draw_board()

        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(RENDER_FPS)

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
        start_x, start_y = TILES_START_X, TILES_START_Y  # Lower number = higher on screen.
        tile_width, tile_height = TILE_WIDTH, TILE_HEIGHT
        col = 0

        for tile_num in range(SMALLEST_TILE, LARGEST_TILE + 1):
            if tiles[tile_num]:  # Only draw available tiles.
                x = start_x + (col % TILES_PER_ROW) * (tile_width + TILE_SPACING)
                y = start_y + (col // TILES_PER_ROW) * (tile_height + TILE_SPACING)

                tile_path = self._sprite_dir.joinpath(f"tile_{tile_num}.png")
                tile_image = pygame.image.load(str(tile_path))
                self._window.blit(tile_image, (x, y))
            col += 1

    def close(self) -> None:
        """Close game."""
        if self._window is not None:
            pygame.quit()
