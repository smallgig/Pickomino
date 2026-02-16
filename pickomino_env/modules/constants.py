"""Project wide constants."""

from __future__ import annotations

__all__ = [
    "ACTION_INDEX_DICE",
    "ACTION_INDEX_ROLL",
    "ACTION_ROLL",
    "ACTION_STOP",
    "ACTION_TUPLE_LENGTH",
    "BACKGROUND_COLOR",
    "BUTTONS_START_X",
    "BUTTONS_START_Y",
    "BUTTON_COLOR",
    "BUTTON_FONT_SIZE",
    "BUTTON_HEIGHT",
    "BUTTON_HOVER_COLOR",
    "BUTTON_SPACING",
    "BUTTON_TEXT_COLOR",
    "BUTTON_WIDTH",
    "DICE_FONT_SIZE",
    "DICE_LABELS_OFFSET_Y",
    "DICE_LABELS_SPACING",
    "DICE_LABEL_COLLECTED",
    "DICE_LABEL_ROLLED",
    "DICE_LABEL_WIDTH",
    "DICE_LABEL_X",
    "DICE_NAMES",
    "DICE_SECTION_START_Y",
    "DICE_SPACING",
    "DIE_SIZE",
    "FONT_COLOR",
    "GREEN",
    "LARGEST_TILE",
    "MAX_BOTS",
    "MIN_ROLLS_FOR_WORM_STRATEGY",
    "NO_GREEN",
    "NO_RED",
    "NUM_DICE",
    "NUM_DIE_FACES",
    "PLAYERS_START_Y",
    "PLAYER_HIGHLIGHT_COLOR",
    "PLAYER_NAME_FONT_SIZE",
    "PLAYER_WIDTH",
    "RED",
    "RENDER_DELAY",
    "RENDER_FPS",
    "RENDER_MODE_HUMAN",
    "RENDER_MODE_RGB_ARRAY",
    "SMALLEST_TILE",
    "TILES_PER_ROW",
    "TILES_ROW_SPACING",
    "TILES_START_X",
    "TILES_START_Y",
    "TILE_HEIGHT",
    "TILE_SPACING",
    "TILE_WIDTH",
    "WINDOW_HEIGHT",
    "WINDOW_WIDTH",
    "WORM_INDEX",
    "WORM_VALUE",
]

from typing import Final

# Coloured printouts.
RED: Final[str] = "\033[31m"
NO_RED: Final[str] = "\033[0m"
GREEN: Final[str] = "\033[32m"
NO_GREEN: Final[str] = "\033[0m"

# Game constants.
SMALLEST_TILE: Final[int] = 21
LARGEST_TILE: Final[int] = 36
NUM_DICE: Final[int] = 8
NUM_DIE_FACES: Final[int] = 6
MAX_BOTS: Final[int] = 6
WORM_INDEX: Final[int] = 5
WORM_VALUE: Final[int] = 5

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
RENDER_DELAY: Final[float] = 0.5

# Rendering window dimensions.
WINDOW_WIDTH: Final[int] = 800
WINDOW_HEIGHT: Final[int] = 600

# Button constants
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 50
BUTTON_SPACING = 20
BUTTONS_START_X = WINDOW_WIDTH - BUTTON_WIDTH - 680
BUTTONS_START_Y = WINDOW_HEIGHT - 180
BUTTON_COLOR = (70, 130, 180)  # Steel blue
BUTTON_HOVER_COLOR = (100, 160, 210)  # Lighter blue
BUTTON_TEXT_COLOR = (255, 255, 255)  # White
BUTTON_FONT_SIZE = 28

# Rendering background color Red, Green, Blue (RGB).
# Muted sage green (100, 140, 100). Dark green (34, 139, 34).
BACKGROUND_COLOR: Final[tuple[int, int, int]] = (70, 130, 70)  # Lighter, softer green.
FONT_COLOR: Final[tuple[int, int, int]] = (0, 0, 0)  # Black

# Player rendering.
PLAYERS_START_Y: Final[int] = 20
PLAYER_NAME_FONT_SIZE: Final[int] = 28
PLAYER_HIGHLIGHT_COLOR: Final[tuple[int, int, int]] = (65, 105, 225)  # Blue
PLAYER_WIDTH: Final[int] = 120

# Dice rendering.
DICE_NAMES: Final[tuple[str, ...]] = (
    "dice_1",
    "dice_2",
    "dice_3",
    "dice_4",
    "dice_5",
    "dice_worm",
)
DICE_SECTION_START_Y: Final[int] = 180
DIE_SIZE: Final[int] = 100
# Horizontal space reserved for left labels (affecting dice positioning).
DICE_LABEL_WIDTH: Final[int] = 100
DICE_SPACING: Final[int] = (WINDOW_WIDTH - DICE_LABEL_WIDTH) // NUM_DIE_FACES

# Dice counts.
DICE_LABEL_COLLECTED: Final[str] = "Collected:"
DICE_LABEL_ROLLED: Final[str] = "Rolled:"
DICE_LABELS_OFFSET_Y: Final[int] = 5  # Distance from dice image bottom to labels.
# Vertical gab between the 'Collected' and 'Rolled' rows.
DICE_LABELS_SPACING: Final[int] = 30
DICE_LABEL_X: Final[int] = 10
DICE_FONT_SIZE: Final[int] = 30

# Tile rendering.
TILE_WIDTH: Final[int] = 60  # 37
TILE_HEIGHT: Final[int] = 100  # 89
TILES_PER_ROW: Final[int] = 8
TILE_SPACING: Final[int] = 10
TILES_ROW_SPACING: Final[int] = 10
TILES_START_X: Final[int] = 150
TILES_START_Y: Final[int] = 380
