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
    ACTION_COLOR,
    ACTION_DISPLAY_X,
    ACTION_DISPLAY_Y,
    ACTION_FONT_SIZE,
    ACTION_ROLL,
    ACTION_STOP,
    ANTIALIAS,
    BACKGROUND_COLOR,
    BUTTON_BORDER_RADIUS,
    BUTTON_BORDER_WIDTH,
    BUTTON_COLOR,
    BUTTON_FONT_SIZE,
    BUTTON_HEIGHT,
    BUTTON_HOVER_COLOR,
    BUTTON_SPACING,
    BUTTON_TEXT_COLOR,
    BUTTON_WIDTH,
    BUTTON_Y_OFFSET,
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
    ERROR_MESSAGE_COLOR,
    FONT_BIG,
    FONT_COLOR,
    FONT_SMALL,
    GAME_OVER_COLOR,
    GAME_OVER_POS,
    LARGEST_TILE,
    NUM_DIE_FACES,
    PLAYER_HIGHLIGHT_COLOR,
    PLAYER_NAME_FONT_SIZE,
    PLAYER_WIDTH,
    PLAYERS_START_Y,
    PLAYERS_TILE_SPACING,
    RENDER_FPS,
    RENDER_MODE_HUMAN,
    RENDER_MODE_RGB_ARRAY,
    SCORES_POS,
    SMALLEST_TILE,
    TILE_HEIGHT,
    TILE_SPACING,
    TILE_WIDTH,
    TILES_HOVER_COLOR,
    TILES_PER_ROW,
    TILES_ROW_SPACING,
    TILES_START_X,
    TILES_START_Y,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    WINNER,
    WINNER_COLOR,
)
from pickomino_env.modules.game import Game

if TYPE_CHECKING:
    from numpy.typing import NDArray  # pyright: ignore[reportUnknownVariableType]

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

        # Error message.
        self._error_message: str | None = None

        # Play again
        self._play_again_yes_rect: pygame.Rect | None = None
        self._play_again_no_rect: pygame.Rect | None = None
        self._play_again: bool | None = None

    def render(self, game: Game) -> NDArray[np.uint8] | None:  # pyright: ignore[reportInvalidTypeForm]
        """Render the environment."""
        self._game.dice = game.dice
        self._game.players = game.players
        self._game.tiles = game.tiles
        self._game.current_player_index = game.current_player_index
        self._game.terminated = game.terminated
        self._error_message = game.explanation if game.truncated else None

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
        if self._game.terminated:
            self._draw_game_over()
        else:
            self._draw_board()

        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(RENDER_FPS)

    def _render_rgb_array(self) -> NDArray[np.uint8]:  # pyright: ignore[reportInvalidTypeForm]
        """Return pixel array for recording."""
        # Like _render_human. But capture as an array.
        if self._window is None:
            pygame.init()
            self._window = pygame.display.set_mode(self._size)
            self._clock = pygame.time.Clock()

        surface = pygame.surfarray.array3d(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            self._window,
        )
        return np.transpose(surface, (1, 0, 2))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    def _handle_mouse_click(self, pos: tuple[int, int]) -> None:
        """Handle mouse click events."""
        if self._roll_button_rect and self._roll_button_rect.collidepoint(pos):
            self._action_click_button = ACTION_ROLL
        elif self._stop_button_rect and self._stop_button_rect.collidepoint(pos):
            self._action_click_button = ACTION_STOP

        # Check dice clicks.
        for index, dice_rect in enumerate(self._dice_rects):
            if dice_rect.collidepoint(pos):
                self._action_click_dice = index
                break

        self._check_play_again(pos)

    def _check_play_again(self, pos: tuple[int, int]) -> None:
        # Check play again.
        if self._play_again_yes_rect and self._play_again_yes_rect.collidepoint(pos):
            self._play_again = True
        if self._play_again_no_rect and self._play_again_no_rect.collidepoint(pos):
            self._play_again = False

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
        """Get the complete action: (dice_index, button_action) or None if incomplete.

        Only used during manual play via main.py.
        """
        if self._action_click_dice is not None and self._action_click_button is not None:
            self._action = (self._action_click_dice, self._action_click_button)
            self._action_click_dice = None
            self._action_click_button = None
            return self._action
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
                tile_x = x + (PLAYER_WIDTH - TILE_WIDTH + PLAYERS_TILE_SPACING) // 2
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
            is_hovered = dice_rect.collidepoint(self._mouse_pos) or self._action_click_dice == index
            if is_hovered:
                highlight_rect = pygame.Rect(x - 3, y - 3, DIE_SIZE + 6, DIE_SIZE + 6)
                pygame.draw.rect(self._window, TILES_HOVER_COLOR, highlight_rect, width=3, border_radius=5)

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
        roll_hovered = self._roll_button_rect.collidepoint(self._mouse_pos) or self._action_click_button == ACTION_ROLL
        stop_hovered = self._stop_button_rect.collidepoint(self._mouse_pos) or self._action_click_button == ACTION_STOP

        # Draw the Roll button.
        roll_color = BUTTON_HOVER_COLOR if roll_hovered else BUTTON_COLOR
        pygame.draw.rect(self._window, roll_color, self._roll_button_rect, border_radius=BUTTON_BORDER_RADIUS)
        pygame.draw.rect(
            self._window,
            FONT_COLOR,
            self._roll_button_rect,
            width=BUTTON_BORDER_WIDTH,
            border_radius=BUTTON_BORDER_RADIUS,
        )

        roll_text = self._button_font.render("ROLL", ANTIALIAS, BUTTON_TEXT_COLOR)
        roll_text_rect = roll_text.get_rect(center=self._roll_button_rect.center)
        self._window.blit(roll_text, roll_text_rect)

        # Draw the Stop button.
        stop_color = BUTTON_HOVER_COLOR if stop_hovered else BUTTON_COLOR
        pygame.draw.rect(self._window, stop_color, self._stop_button_rect, border_radius=BUTTON_BORDER_RADIUS)
        pygame.draw.rect(
            self._window,
            FONT_COLOR,
            self._stop_button_rect,
            width=BUTTON_BORDER_WIDTH,
            border_radius=BUTTON_BORDER_RADIUS,
        )

        stop_text = self._button_font.render("STOP", ANTIALIAS, BUTTON_TEXT_COLOR)
        stop_text_rect = stop_text.get_rect(center=self._stop_button_rect.center)
        self._window.blit(stop_text, stop_text_rect)

    def _draw_action_display(self) -> None:
        """Draw the current action selection. Only active during manual play via main.py."""
        if self._action is not None and self._window is not None:
            dice_idx, button_action = self._action
            font = pygame.font.SysFont(None, ACTION_FONT_SIZE)
            text = f"Action: ({dice_idx}, {button_action})"
            surface = font.render(text, ANTIALIAS, ACTION_COLOR)
            self._window.blit(surface, (ACTION_DISPLAY_X, ACTION_DISPLAY_Y))

    def _draw_error_message(self) -> None:
        """Draw the error message. Only active if the last action is invalid."""
        if self._error_message is None or self._window is None:
            return
        font = pygame.font.SysFont(None, ACTION_FONT_SIZE)
        surface = font.render(f"Error: {self._error_message}", ANTIALIAS, ERROR_MESSAGE_COLOR)
        self._window.blit(surface, (ACTION_DISPLAY_X, ACTION_DISPLAY_Y + ACTION_FONT_SIZE + 5))

    def _show_score(self, font_small: pygame.font.Font) -> None:
        """Show the scores on the game over screen."""
        if self._window is None:
            return

        # Scores
        player_score = self._game.players[0].end_score()
        bot_scores: list[int] = [p.end_score() for p in self._game.players[1:]]

        score_text = f"Your Score: {player_score}   | Bots: {bot_scores}"
        surf = font_small.render(score_text, ANTIALIAS, FONT_COLOR)
        x, y = GAME_OVER_POS
        self._window.blit(surf, surf.get_rect(center=(x, y + SCORES_POS)))

    def _determine_winner(self, font_small: pygame.font.Font) -> None:
        """Determine the winner at the end of the game."""
        if self._window is None:
            return

        # Scores
        all_scores: list[int] = [p.end_score() for p in self._game.players]
        max_score = max(all_scores)

        # Among players with the highest score, pick the one with the highest tile.
        candidates = [p for p in self._game.players if p.end_score() == max_score]
        winner = max(candidates, key=lambda p: p.highest_tile())

        x, y = GAME_OVER_POS
        winner_text = f"Winner: {winner.name} with {max_score} points"
        winner_surf = font_small.render(winner_text, ANTIALIAS, WINNER_COLOR)
        self._window.blit(winner_surf, winner_surf.get_rect(center=(x, y + SCORES_POS + WINNER)))

    def _draw_play_again(self) -> None:
        """Draw play again in the game over screen."""
        if self._window is None:
            return
        font = pygame.font.SysFont(None, BUTTON_FONT_SIZE)
        center_x, center_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2

        question = font.render("Play again?", ANTIALIAS, FONT_COLOR)
        self._window.blit(question, question.get_rect(center=(center_x, center_y)))

        self._play_again_yes_rect = pygame.Rect(
            center_x - BUTTON_WIDTH - BUTTON_SPACING // 2, center_y + BUTTON_Y_OFFSET, BUTTON_WIDTH, BUTTON_HEIGHT
        )
        self._play_again_no_rect = pygame.Rect(
            center_x + BUTTON_SPACING // 2, center_y + BUTTON_Y_OFFSET, BUTTON_WIDTH, BUTTON_HEIGHT
        )

        for rect, label in ((self._play_again_yes_rect, "YES"), (self._play_again_no_rect, "NO")):
            hovered = rect.collidepoint(self._mouse_pos)
            pygame.draw.rect(
                self._window, BUTTON_HOVER_COLOR if hovered else BUTTON_COLOR, rect, border_radius=BUTTON_BORDER_RADIUS
            )
            pygame.draw.rect(
                self._window, FONT_COLOR, rect, width=BUTTON_BORDER_WIDTH, border_radius=BUTTON_BORDER_RADIUS
            )
            self._window.blit(
                font.render(label, ANTIALIAS, BUTTON_TEXT_COLOR),
                font.render(label, ANTIALIAS, BUTTON_TEXT_COLOR).get_rect(center=rect.center),
            )

    def _draw_game_over(self) -> None:
        """Draw the game over screen when the game is terminated."""
        if self._window is None:
            return
        self._window.fill(BACKGROUND_COLOR)
        font_big = pygame.font.SysFont(None, FONT_BIG)
        font_small = pygame.font.SysFont(None, FONT_SMALL)

        surface = font_big.render("GAME OVER", ANTIALIAS, GAME_OVER_COLOR)
        rect = surface.get_rect(center=GAME_OVER_POS)
        self._window.blit(surface, rect)

        self._show_score(font_small)
        self._determine_winner(font_small)
        self._draw_play_again()

    def get_play_again(self) -> bool | None:
        """Return play again for the game loop."""
        return self._play_again

    def _draw_board(self) -> None:
        """Draw the game board with tiles and dice."""
        if self._window is None:
            return

        self._draw_players()
        self._draw_dice()
        self._draw_tiles()
        self._draw_action_display()
        self._draw_error_message()
        self._draw_buttons()

    def close(self) -> None:
        """Close game."""
        if self._window is not None:
            pygame.quit()
