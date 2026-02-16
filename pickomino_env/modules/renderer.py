"""Renderer using pygame."""

from __future__ import annotations

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files  # type: ignore[import-not-found, no-redef]

from typing import TYPE_CHECKING

import numpy as np
import pygame  # Uses pygame-ce, see pyproject.toml

from pickomino_env.modules.constants import (
    BACKGROUND_COLOR,
    BUTTON_COLOR,
    BUTTON_FONT_SIZE,
    BUTTON_HEIGHT,
    BUTTON_HOVER_COLOR,
    BUTTON_SPACING,
    BUTTON_TEXT_COLOR,
    BUTTON_WIDTH,
    BUTTONS_START_X,
    BUTTONS_START_Y,
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

if TYPE_CHECKING:
    from numpy.typing import NDArray  # pyright: ignore[reportUnknownVariableType]

    from pickomino_env.modules.dice import Dice
    from pickomino_env.modules.player import Player
    from pickomino_env.modules.tiles import Tiles


__all__ = ["Renderer"]


# pylint: disable=too-many-instance-attributes
class Renderer:
    """Class Renderer."""

    def __init__(self, render_mode: str | None = None) -> None:
        """Initialize Renderer."""
        self._render_mode: str | None = render_mode
        self._window: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None

        self._action: tuple[int, int] | None = None
        self._action_click_button: int | None = None  # 0 for a roll, 1 for a stop.
        self._action_click_dice: int | None = None  # 0-5 for the die faces.

        # Screen size
        self._size: tuple[int, int] = (WINDOW_WIDTH, WINDOW_HEIGHT)

        self._game: Game = Game()
        self._sprite_dir = files("pickomino_env").joinpath("sprites")
        # Lazy initialization: pygame not initialized during __init__(), so create font on the first render.
        self._dice_font: pygame.font.Font | None = None
        self._button_font: pygame.font.Font | None = None

        # Button rectangles.
        self._roll_button_rect: pygame.Rect | None = None
        self._stop_button_rect: pygame.Rect | None = None

        # Dice rectangles.
        self._dice_rects: list[pygame.Rect] = []

        self._mouse_pos: tuple[int, int] = (0, 0)

    def render(
        self,
        dice: Dice,
        players: list[Player],
        tiles: Tiles,
        current_player_index: int,
    ) -> NDArray[np.uint8] | None:  # pyright: ignore[reportInvalidTypeForm]
        """Render the environment."""
        self._game.dice = dice
        self._game.players = players
        self._game.tiles = tiles
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
            if event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_click(event.pos)

        self._mouse_pos = pygame.mouse.get_pos()

        # Draw background.
        self._window.fill(BACKGROUND_COLOR)  # Lighter, softer green.

        # Draw game state.
        self._draw_board()

        # Draw buttons.
        self._draw_buttons()

        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(RENDER_FPS)

    def _render_rgb_array(self) -> NDArray[np.uint8]:  # pyright: ignore[reportInvalidTypeForm]
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

    def _handle_mouse_click(self, pos: tuple[int, int]) -> None:
        """Handle mouse click events."""
        if self._roll_button_rect and self._roll_button_rect.collidepoint(pos):
            self._action_click_button = 0  # Roll
        elif self._stop_button_rect and self._stop_button_rect.collidepoint(pos):
            self._action_click_button = 1  # Stop

        # Check dice clicks.
        for index, dice_rect in enumerate(self._dice_rects):
            if dice_rect.collidepoint(pos):
                self._action_click_dice = index
                break

    def get_action(self) -> int | None:
        """Get the selected action and reset it."""
        action = self._action_click_button
        self._action_click_button = None
        return action

    def get_dice_selection(self) -> int | None:
        """Get the selected dice index (0-5) and reset it."""
        dice_index = self._action_click_dice
        self._action_click_dice = None
        return dice_index

    def get_full_action(self) -> tuple[int, int] | None:
        """Get complete action: (button_action, dice_index) or None if incomplete."""
        if self._action_click_button is not None and self._action_click_dice is not None:
            action = (self._action_click_dice, self._action_click_button)
            self._action_click_button = None
            self._action_click_dice = None
            return action
        return None

    def _draw_players(self) -> None:
        """Draw player names and their top tile."""
        if self._window is None:
            return

        font = pygame.font.SysFont(None, PLAYER_NAME_FONT_SIZE)

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
                tile_path = self._sprite_dir.joinpath(  # pyright: ignore[reportUnknownVariableType]
                    f"tile_{current_tile}.png",
                )
                tile_image = pygame.image.load(str(tile_path))
                tile_x = x + (PLAYER_WIDTH - TILE_WIDTH) // 2
                tile_y = PLAYERS_START_Y + PLAYER_NAME_FONT_SIZE
                self._window.blit(tile_image, (tile_x, tile_y))

    def _draw_dice(self) -> None:
        """Draw the dice section with counts."""
        if self._window is None:
            return

        # Lazy initialization: pygame not initialized during __init__(), so create font on rendering.
        self._dice_font = pygame.font.SysFont(None, DICE_FONT_SIZE)
        # Reset dice rectangles
        self._dice_rects = []

        y: int = DICE_SECTION_START_Y

        # Draw dice images
        for index, die_name in enumerate(DICE_NAMES):
            x: int = DICE_LABEL_WIDTH + index * DICE_SPACING + (DICE_SPACING - DIE_SIZE) // 2
            die_path = self._sprite_dir.joinpath(f"{die_name}.png")  # pyright: ignore[reportUnknownVariableType]
            die_image = pygame.image.load(str(die_path))
            die_image = pygame.transform.scale(die_image, (DIE_SIZE, DIE_SIZE))
            self._window.blit(die_image, (x, y))

            dice_rect = pygame.Rect(x, y, DIE_SIZE, DIE_SIZE)
            self._dice_rects.append(dice_rect)

            # Hover effect
            is_hovered = dice_rect.collidepoint(self._mouse_pos)
            if is_hovered:
                highlight_rect = pygame.Rect(x - 3, y - 3, DIE_SIZE + 6, DIE_SIZE + 6)
                pygame.draw.rect(self._window, (255, 255, 0), highlight_rect, width=3, border_radius=5)

        self._draw_dice_counts(0)  # Collected.
        self._draw_dice_counts(1)  # Rolled.

        # Score label
        score_y = DICE_SECTION_START_Y + DIE_SIZE + DICE_LABELS_OFFSET_Y + 2 * DICE_LABELS_SPACING
        score_text = f"Score: {self._game.dice.score()[0]}"
        score_text_surface = True
        score_surface = self._dice_font.render(score_text, score_text_surface, FONT_COLOR)
        self._window.blit(score_surface, (DICE_LABEL_X, score_y))

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

        label_surface_text = True
        label_surface = self._dice_font.render(
            label,
            label_surface_text,
            FONT_COLOR,
        )
        self._window.blit(label_surface, (DICE_LABEL_X, labels_y))

        # Draw counts.
        for index in range(NUM_DIE_FACES):
            x = DICE_LABEL_WIDTH + index * DICE_SPACING + (DICE_SPACING - DIE_SIZE) // 2
            count_surface_text = True
            count_text = self._dice_font.render(
                str(counts[index]),
                count_surface_text,
                FONT_COLOR,
            )
            text_width = count_text.get_width()
            count_text_x = x + (DIE_SIZE - text_width) // 2
            self._window.blit(count_text, (count_text_x, labels_y))

    def _draw_tiles(self) -> None:
        """Draw table tiles (21-36 in a grid at bottom)."""
        if self._window is None:
            return

        tiles = self._game.tiles.get_tiles()

        for col, tile_num in enumerate(range(SMALLEST_TILE, LARGEST_TILE + 1)):
            if tiles[tile_num]:  # Only draw available tiles.
                x = TILES_START_X + (col % TILES_PER_ROW) * (TILE_WIDTH + TILE_SPACING)
                y = TILES_START_Y + (col // TILES_PER_ROW) * (TILE_HEIGHT + TILES_ROW_SPACING)

                tile_path = self._sprite_dir.joinpath(  # pyright: ignore[reportUnknownVariableType]
                    f"tile_{tile_num}.png",
                )
                tile_image = pygame.image.load(str(tile_path))
                self._window.blit(tile_image, (x, y))

    def _draw_buttons(self) -> None:
        """Draw the Roll and Stop buttons."""
        if self._window is None:
            return

        # Lazy initialization of button font.
        if self._button_font is None:
            self._button_font = pygame.font.SysFont(None, BUTTON_FONT_SIZE)

        # Roll button.
        roll_x = BUTTONS_START_X
        roll_y = BUTTONS_START_Y
        self._roll_button_rect = pygame.Rect(roll_x, roll_y, BUTTON_WIDTH, BUTTON_HEIGHT)

        # Stop button.
        stop_x = BUTTONS_START_X
        stop_y = BUTTONS_START_Y + BUTTON_HEIGHT + BUTTON_SPACING
        self._stop_button_rect = pygame.Rect(stop_x, stop_y, BUTTON_WIDTH, BUTTON_HEIGHT)

        # Check the hover state.
        roll_hovered = self._roll_button_rect.collidepoint(self._mouse_pos)
        stop_hovered = self._stop_button_rect.collidepoint(self._mouse_pos)

        # Draw the Roll button.
        roll_color = BUTTON_HOVER_COLOR if roll_hovered else BUTTON_COLOR
        pygame.draw.rect(self._window, roll_color, self._roll_button_rect, border_radius=10)
        pygame.draw.rect(self._window, FONT_COLOR, self._roll_button_rect, width=2, border_radius=10)

        antialias = True

        roll_text = self._button_font.render("ROLL", antialias, BUTTON_TEXT_COLOR)
        roll_text_rect = roll_text.get_rect(center=self._roll_button_rect.center)
        self._window.blit(roll_text, roll_text_rect)

        # Draw the Stop button.
        stop_color = BUTTON_HOVER_COLOR if stop_hovered else BUTTON_COLOR
        pygame.draw.rect(self._window, stop_color, self._stop_button_rect, border_radius=10)
        pygame.draw.rect(self._window, FONT_COLOR, self._stop_button_rect, width=2, border_radius=10)

        stop_text = self._button_font.render("STOP", antialias, BUTTON_TEXT_COLOR)
        stop_text_rect = stop_text.get_rect(center=self._stop_button_rect.center)
        self._window.blit(stop_text, stop_text_rect)

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
