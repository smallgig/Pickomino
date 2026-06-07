"""Project wide constants."""

from __future__ import annotations

__all__ = [
    "ACTION_COLOR",
    "ACTION_DISPLAY_X",
    "ACTION_DISPLAY_Y",
    "ACTION_FONT_SIZE",
    "ACTION_INDEX_DICE",
    "ACTION_INDEX_ROLL",
    "ACTION_ROLL",
    "ACTION_STOP",
    "ACTION_TUPLE_LENGTH",
    "ANTIALIAS",
    "BACKGROUND_COLOR",
    "BOT_NAMES",
    "BUTTONS_START_X",
    "BUTTONS_START_Y",
    "BUTTON_BORDER_RADIUS",
    "BUTTON_BORDER_WIDTH",
    "BUTTON_COLOR",
    "BUTTON_FONT_SIZE",
    "BUTTON_HEIGHT",
    "BUTTON_HOVER_COLOR",
    "BUTTON_SPACING",
    "BUTTON_TEXT_COLOR",
    "BUTTON_WIDTH",
    "BUTTON_Y_OFFSET",
    "DICE_COUNT_ROW_COLLECTED",
    "DICE_COUNT_ROW_ROLLED",
    "DICE_FONT_SIZE",
    "DICE_HOVER_BORDER_RADIUS",
    "DICE_HOVER_BORDER_WIDTH",
    "DICE_HOVER_OFFSET",
    "DICE_LABELS_OFFSET_Y",
    "DICE_LABELS_SPACING",
    "DICE_LABEL_COLLECTED",
    "DICE_LABEL_ROLLED",
    "DICE_LABEL_WIDTH",
    "DICE_LABEL_X",
    "DICE_NAMES",
    "DICE_SCORE_ROW_INDEX",
    "DICE_SECTION_START_Y",
    "DICE_SPACING",
    "DIE_SIZE",
    "ERROR_MESSAGE_COLOR",
    "FONT_BIG",
    "FONT_COLOR",
    "FONT_SMALL",
    "GAME_OVER_COLOR",
    "GAME_OVER_POS",
    "LARGEST_TILE",
    "MAX_BOTS",
    "MIN_ROLLS_FOR_WORM_STRATEGY",
    "NUM_DICE",
    "NUM_DIE_FACES",
    "PLAYERS_START_Y",
    "PLAYERS_TILE_SPACING",
    "PLAYER_HIGHLIGHT_COLOR",
    "PLAYER_NAME_FONT_SIZE",
    "PLAYER_WIDTH",
    "RENDER_DELAY",
    "RENDER_FPS",
    "RENDER_MODE_HUMAN",
    "RENDER_MODE_RGB_ARRAY",
    "SCORES_POS",
    "SMALLEST_TILE",
    "TILES_HOVER_COLOR",
    "TILES_PER_ROW",
    "TILES_ROW_SPACING",
    "TILES_START_X",
    "TILES_START_Y",
    "TILE_HEIGHT",
    "TILE_SPACING",
    "TILE_WIDTH",
    "WINDOW_HEIGHT",
    "WINDOW_WIDTH",
    "WINNER",
    "WINNER_COLOR",
    "WORM_INDEX",
    "WORM_VALUE",
]

from typing import Final

# Game constants.
SMALLEST_TILE: Final[int] = 21
LARGEST_TILE: Final[int] = 36
NUM_DICE: Final[int] = 8
NUM_DIE_FACES: Final[int] = 6
MAX_BOTS: Final[int] = 6
WORM_INDEX: Final[int] = 5
WORM_VALUE: Final[int] = 5
BOT_NAMES: Final[tuple[str, ...]] = ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot")

# Action constants.
ACTION_INDEX_DICE: Final[int] = 0
ACTION_INDEX_ROLL: Final[int] = 1
ACTION_ROLL: Final[int] = 0
ACTION_STOP: Final[int] = 1
ACTION_TUPLE_LENGTH: Final[int] = 2

# Bot strategy.
MIN_ROLLS_FOR_WORM_STRATEGY: Final[int] = 3

# Rendering mode constants.
RENDER_MODE_HUMAN: Final[str] = "human"
RENDER_MODE_RGB_ARRAY: Final[str] = "rgb_array"

# Rendering frequency and delay for bots.
RENDER_FPS: Final[int] = 60  # Frames Per Second.
RENDER_DELAY: Final[float] = 0

# Rendering window dimensions.
WINDOW_WIDTH: Final[int] = 1000
WINDOW_HEIGHT: Final[int] = 750

# Game Over alignment
GAME_OVER_POS: Final[tuple[int, int]] = (500, 150)
GAME_OVER_COLOR: Final[tuple[int, int, int]] = (200, 0, 0)
SCORES_POS: Final[int] = 80
FONT_BIG: Final[int] = 80
FONT_SMALL: Final[int] = 40

# Error Message
ERROR_MESSAGE_COLOR: Final[tuple[int, int, int]] = (200, 0, 0)

# Winner alignment
WINNER_COLOR: Final[tuple[int, int, int]] = (255, 215, 0)  # Gold
WINNER: Final[int] = 60

# Button constants
BUTTON_WIDTH: Final[int] = 120
BUTTON_HEIGHT: Final[int] = 50
BUTTON_SPACING: Final[int] = 20
BUTTONS_START_X: Final[int] = WINDOW_WIDTH - BUTTON_WIDTH - 870
BUTTONS_START_Y: Final[int] = WINDOW_HEIGHT - 270
BUTTON_COLOR: Final[tuple[int, int, int]] = (70, 130, 180)  # Steel blue
BUTTON_HOVER_COLOR: Final[tuple[int, int, int]] = (100, 160, 210)  # Lighter blue
BUTTON_TEXT_COLOR: Final[tuple[int, int, int]] = (255, 255, 255)  # White
BUTTON_FONT_SIZE: Final[int] = 28
BUTTON_BORDER_RADIUS: Final[int] = 10
BUTTON_BORDER_WIDTH: Final[int] = 2
BUTTON_Y_OFFSET: Final[int] = 40


# Rendering background color Red, Green, Blue (RGB).
# Muted sage green (100, 140, 100). Dark green (34, 139, 34).
BACKGROUND_COLOR: Final[tuple[int, int, int]] = (70, 130, 70)  # Lighter, softer green.
FONT_COLOR: Final[tuple[int, int, int]] = (0, 0, 0)  # Black

# Player rendering.
PLAYERS_START_Y: Final[int] = 20
PLAYER_NAME_FONT_SIZE: Final[int] = 28
PLAYER_HIGHLIGHT_COLOR: Final[tuple[int, int, int]] = (65, 105, 225)  # Blue
PLAYER_WIDTH: Final[int] = 120
PLAYERS_TILE_SPACING: Final[int] = 50

# Dice rendering.
DICE_NAMES: Final[tuple[str, ...]] = (
    "dice_1",
    "dice_2",
    "dice_3",
    "dice_4",
    "dice_5",
    "dice_worm",
)
DICE_SECTION_START_Y: Final[int] = 250
DIE_SIZE: Final[int] = 100
# Horizontal space reserved for left labels (affecting dice positioning).
DICE_LABEL_WIDTH: Final[int] = 100
DICE_SPACING: Final[int] = (WINDOW_WIDTH - DICE_LABEL_WIDTH) // NUM_DIE_FACES

# Dice counts.
DICE_COUNT_ROW_COLLECTED: Final[int] = 0
DICE_COUNT_ROW_ROLLED: Final[int] = 1
DICE_SCORE_ROW_INDEX: Final[int] = 2
DICE_LABEL_COLLECTED: Final[str] = "Collected:"
DICE_LABEL_ROLLED: Final[str] = "Rolled:"
DICE_LABELS_OFFSET_Y: Final[int] = 5  # Distance from dice image bottom to labels.
# Vertical gab between the 'Collected' and 'Rolled' rows.
DICE_LABELS_SPACING: Final[int] = 30
DICE_LABEL_X: Final[int] = 10
DICE_FONT_SIZE: Final[int] = 30

# Tile rendering.
TILE_WIDTH: Final[int] = 100
TILE_HEIGHT: Final[int] = 140
TILES_PER_ROW: Final[int] = 8
TILE_SPACING: Final[int] = 8
TILES_ROW_SPACING: Final[int] = -15
TILES_START_X: Final[int] = 170
TILES_START_Y: Final[int] = 480
TILES_HOVER_COLOR: Final[tuple[int, int, int]] = (255, 255, 0)
DICE_HOVER_OFFSET: Final[int] = 3
DICE_HOVER_BORDER_WIDTH: Final[int] = 3
DICE_HOVER_BORDER_RADIUS: Final[int] = 5

# Display action.
ACTION_DISPLAY_X: Final[int] = 20
ACTION_DISPLAY_Y: Final[int] = DICE_SECTION_START_Y - 100
ACTION_FONT_SIZE: Final[int] = 28
ACTION_COLOR: Final[tuple[int, int, int]] = (0, 0, 0)  # Black

ANTIALIAS: Final[bool] = True
