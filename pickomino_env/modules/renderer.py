"""Renderer using pygame."""

from __future__ import annotations

__all__ = ["Renderer"]
# pygame internally uses deprecated pkg_resources
# See: https://setuptools.pypa.io/en/latest/pkg_resources.html
import warnings
from importlib.resources import files

import numpy as np

from pickomino_env.modules.constants import (
    BACKGROUND_COLOR,
    DICE_FONT_SIZE,
    DICE_LABEL_COLLECTED,
    DICE_LABEL_ROLLED,
    DICE_LABEL_WIDTH,
    DICE_LABEL_X,
    DICE_LABELS_OFFSET_Y,
    DICE_LABELS_SPACING,
    DICE_NAMES,
    DICE_SECTION_START_Y,
    DICE_SPACING,
    DIE_SIZE,
    FONT_COLOR,
    LARGEST_TILE,
    NUM_DIE_FACES,
    PLAYER_HIGHLIGHT_COLOR,
    PLAYER_NAME_FONT_SIZE,
    PLAYER_WIDTH,
    PLAYERS_START_Y,
    RENDER_FPS,
    RENDER_MODE_HUMAN,
    RENDER_MODE_RGB_ARRAY,
    SMALLEST_TILE,
    TILE_HEIGHT,
    TILE_SPACING,
    TILE_WIDTH,
    TILES_PER_ROW,
    TILES_ROW_SPACING,
    TILES_START_X,
    TILES_START_Y,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from pickomino_env.modules.game import Game

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
# E402: module level import not at the top of the file. Needed to suppress warning before import.
import pygame  # noqa: RUF100, E402 # pylint: disable=wrong-import-position, wrong-import-order


class Renderer:
    """Class Renderer."""

    def __init__(self, render_mode: str | None = None) -> None:
        """Initialize Renderer."""
        self._render_mode: str | None = render_mode
        self._window: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None

        # Screen size
        self._size: tuple[int, int] = (WINDOW_WIDTH, WINDOW_HEIGHT)

        self._game: Game = Game()
        self._sprite_dir = files("pickomino_env").joinpath("sprites")
        # Lazy initialization: pygame not initialized during __init__(), so create font on the first render.
        self._dice_font: pygame.font.Font | None = None

    def render(
        self,
        dice: Game.Dice,
        players: list[Game.Player],
        tiles: Game.TableTiles,
        current_player_index: int,
    ) -> np.ndarray | list[np.ndarray] | None:
        """Render the environment."""
        self._game.dice = dice
        self._game.players = players
        self._game.table_tiles = tiles
        self._game.current_player_index = current_player_index

        if self._render_mode is None:
            return None

        if self._render_mode == RENDER_MODE_HUMAN:
            self._render_human()
        elif self._render_mode == RENDER_MODE_RGB_ARRAY:
            return self._render_rgb_array()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return None

    def _render_human(self) -> None:
        """Display to screen."""
        if self._window is None:
            pygame.init()
            self._window = pygame.display.set_mode(self._size)
            self._clock = pygame.time.Clock()

        # Check for window close event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

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
            raise RuntimeError(  # noqa: RUF100, TRY003 message variable unnecessary.
                "Window not initialised.",
            )
        surface = pygame.surfarray.array3d(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            self._window,
        )
        return np.transpose(surface, (1, 0, 2))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    def _draw_players(self) -> None:
        """Draw player names and their top tile."""
        if self._window is None:
            return

        font = pygame.font.Font(None, PLAYER_NAME_FONT_SIZE)

        for index, player in enumerate(self._game.players):
            x = index * PLAYER_WIDTH

            # Highlight background for the current player
            if index == self._game.current_player_index:
                pygame.draw.rect(
                    self._window,
                    PLAYER_HIGHLIGHT_COLOR,
                    (x, PLAYERS_START_Y, PLAYER_WIDTH, PLAYER_NAME_FONT_SIZE),
                )

            # Draw name
            name_surface = font.render(player.name, True, FONT_COLOR)  # noqa: RUF100, FBT003 API constraint.
            name_x = x + (PLAYER_WIDTH - name_surface.get_width()) // 2
            self._window.blit(name_surface, (name_x, PLAYERS_START_Y + 5))

            # Draw tile sprite
            current_tile = player.show()
            if current_tile > 0:
                tile_path = self._sprite_dir.joinpath(f"tile_{current_tile}.png")
                tile_image = pygame.image.load(str(tile_path))
                tile_x = x + (PLAYER_WIDTH - TILE_WIDTH) // 2
                tile_y = PLAYERS_START_Y + PLAYER_NAME_FONT_SIZE
                self._window.blit(tile_image, (tile_x, tile_y))

    def _draw_dice(self) -> None:
        """Draw the dice section with counts."""
        if self._window is None:
            return

        # Lazy initialization: pygame not initialized during __init__(), so create font on rendering.
        self._dice_font = pygame.font.Font(None, DICE_FONT_SIZE)

        y: int = DICE_SECTION_START_Y

        # Draw dice images
        for index, die_name in enumerate(DICE_NAMES):
            x: int = DICE_LABEL_WIDTH + index * DICE_SPACING + (DICE_SPACING - DIE_SIZE) // 2
            die_path = self._sprite_dir.joinpath(f"{die_name}.png")
            die_image = pygame.image.load(str(die_path))
            die_image = pygame.transform.scale(die_image, (DIE_SIZE, DIE_SIZE))
            self._window.blit(die_image, (x, y))

        self._draw_dice_counts(0)  # Collected.
        self._draw_dice_counts(1)  # Rolled.

    def _draw_dice_counts(self, row_index: int) -> None:
        """Draw the label and counts."""
        if self._window is None or self._dice_font is None:
            return

        # Draw labels.
        labels_y: int = DICE_SECTION_START_Y + DIE_SIZE + DICE_LABELS_OFFSET_Y + row_index * DICE_LABELS_SPACING
        if row_index == 0:  # Collected.
            label: str = DICE_LABEL_COLLECTED
            counts: list[int] = self._game.dice.get_collected()
        else:  # Rolled.
            label = DICE_LABEL_ROLLED
            counts = self._game.dice.get_rolled()

        label_surface = self._dice_font.render(label, True, FONT_COLOR)  # noqa: RUF100, FBT003 API constraint.
        self._window.blit(label_surface, (DICE_LABEL_X, labels_y))

        # Draw counts.
        for index in range(NUM_DIE_FACES):
            x = DICE_LABEL_WIDTH + index * DICE_SPACING + (DICE_SPACING - DIE_SIZE) // 2
            count_text = self._dice_font.render(
                str(counts[index]),
                True,  # noqa: RUF100, FBT003 API constraint.
                FONT_COLOR,
            )
            text_width = count_text.get_width()
            count_text_x = x + (DIE_SIZE - text_width) // 2
            self._window.blit(count_text, (count_text_x, labels_y))

    def _draw_tiles(self) -> None:
        """Draw table tiles (21-36 in a grid at bottom)."""
        if self._window is None:
            return

        tiles = self._game.table_tiles.get_table()

        for col, tile_num in enumerate(range(SMALLEST_TILE, LARGEST_TILE + 1)):
            if tiles[tile_num]:  # Only draw available tiles.
                x = TILES_START_X + (col % TILES_PER_ROW) * (TILE_WIDTH + TILE_SPACING)
                y = TILES_START_Y + (col // TILES_PER_ROW) * (TILE_HEIGHT + TILES_ROW_SPACING)

                tile_path = self._sprite_dir.joinpath(f"tile_{tile_num}.png")
                tile_image = pygame.image.load(str(tile_path))
                self._window.blit(tile_image, (x, y))

    def _draw_board(self) -> None:
        """Draw the game board with tiles and dice."""
        if self._window is None:
            return

        self._draw_players()
        self._draw_dice()
        self._draw_tiles()

    def close(self) -> None:
        """Close game."""
        if self._window is not None:
            pygame.quit()
