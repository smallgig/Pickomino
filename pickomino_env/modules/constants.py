"""Project wide constants."""

__all__ = [
    "RED",
    "NO_RED",
    "GREEN",
    "NO_GREEN",
    "SMALLEST_TILE",
    "LARGEST_TILE",
    "NUM_DICE",
    "NUM_DIE_FACES",
    "ACTION_ROLL",
    "ACTION_STOP",
    "ACTION_INDEX_ROLL",
    "ACTION_INDEX_DICE",
    "RENDER_MODE_HUMAN",
    "RENDER_MODE_RGB_ARRAY",
    "RENDER_FPS",
    "RENDER_DELAY",
    "WINDOW_WIDTH",
    "WINDOW_HEIGHT",
    "BACKGROUND_COLOR",
    "FONT_COLOR",
    "PLAYERS_START_Y",
    "PLAYER_NAME_FONT_SIZE",
    "PLAYER_HIGHLIGHT_COLOR",
    "PLAYER_WIDTH",
    "DICE_NAMES",
    "DICE_SECTION_START_Y",
    "DIE_SIZE",
    "DICE_LABEL_WIDTH",
    "DICE_SPACING",
    "DICE_LABEL_COLLECTED",
    "DICE_LABEL_ROLLED",
    "DICE_LABELS_OFFSET_Y",
    "DICE_LABELS_SPACING",
    "DICE_LABEL_X",
    "DICE_FONT_SIZE",
    "TILE_WIDTH",
    "TILE_HEIGHT",
    "TILES_PER_ROW",
    "TILE_SPACING",
    "TILES_ROW_SPACING",
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
NUM_DIE_FACES: Final[int] = 6

# Action constants.
ACTION_INDEX_DICE: Final[int] = 0
ACTION_INDEX_ROLL: Final[int] = 1
ACTION_ROLL: Final[int] = 0
ACTION_STOP: Final[int] = 1

# Rendering mode constants.
RENDER_MODE_HUMAN: Final[str] = "human"
RENDER_MODE_RGB_ARRAY: Final[str] = "rgb_array"

# Rendering frequency and delay for bots.
RENDER_FPS: Final[int] = 60  # Frames Per Second.
RENDER_DELAY: Final[float] = 0.5

# Rendering window dimensions.
WINDOW_WIDTH: Final[int] = 800
WINDOW_HEIGHT: Final[int] = 600

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
