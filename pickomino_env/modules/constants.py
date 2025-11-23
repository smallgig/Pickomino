"""Project wide constants."""

__all__ = [
    "RED",
    "NO_RED",
    "GREEN",
    "NO_GREEN",
    "SMALLEST_TILE",
    "LARGEST_TILE",
    "NUM_DICE",
    "ACTION_ROLL",
    "ACTION_STOP",
    "ACTION_INDEX_ROLL",
    "ACTION_INDEX_DICE",
    "RENDER_MODE_HUMAN",
    "RENDER_MODE_RGB_ARRAY",
    "WINDOW_WIDTH",
    "WINDOW_HEIGHT",
    "BACKGROUND_COLOR",
    "TILE_WIDTH",
    "TILE_HEIGHT",
    "TILES_PER_ROW",
    "TILE_SPACING",
    "TILES_START_X",
    "TILES_START_Y",
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

# Action constants.
ACTION_INDEX_DICE: Final[int] = 0
ACTION_INDEX_ROLL: Final[int] = 1
ACTION_ROLL: Final[int] = 0
ACTION_STOP: Final[int] = 1

# Rendering mode constants.
RENDER_MODE_HUMAN: Final[str] = "human"
RENDER_MODE_RGB_ARRAY: Final[str] = "rgb_array"

# Rendering window dimensions.
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Rendering background color (RGB).
BACKGROUND_COLOR = (70, 130, 70)  # Nice green.

# Tile rendering
TILE_WIDTH = 80
TILE_HEIGHT = 160
TILES_PER_ROW = 8
TILE_SPACING = 10
TILES_START_X = 50
TILES_START_Y = 250
