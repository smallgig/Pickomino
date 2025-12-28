"""Provide a method to check that an action is allowed."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from pickomino_env.modules.constants import (
    ACTION_INDEX_DICE,
    ACTION_INDEX_ROLL,
    ACTION_ROLL,
    ACTION_STOP,
    ACTION_TUPLE_LENGTH,
    NO_RED,
    NUM_DIE_FACES,
    RED,
)

if TYPE_CHECKING:
    from pickomino_env.modules.dice import Dice

__all__ = ["ActionChecker"]


class ActionChecker:
    """Class ActionChecker."""

    def __init__(self, dice: Dice) -> None:
        """Initialize ActionChecker."""
        self._action: tuple[object, object] = (0, 0)  # Dummy initialisation.
        self._terminated = False
        self._truncated = False
        self._explanation = ""
        self._dice = dice

    def _validate_dice_index(self) -> None:
        """Validate action[0] (dice_index)."""
        dice_index = self._action[ACTION_INDEX_DICE]
        if not isinstance(dice_index, (int, np.integer)):
            raise TypeError(f"action[0] must be an integer, got {type(dice_index).__name__}")
        if dice_index < 0 or dice_index > NUM_DIE_FACES - 1:
            raise ValueError(f"action[0] must be 0-{NUM_DIE_FACES - 1}, got {dice_index}")

    def _validate_roll_action(self) -> None:
        """Validate action[1] (roll action)."""
        roll_action = self._action[ACTION_INDEX_ROLL]
        if not isinstance(roll_action, (int, np.integer)):
            raise TypeError(f"action[1] must be an integer, got {type(roll_action).__name__}")
        if roll_action not in {ACTION_STOP, ACTION_ROLL}:
            raise ValueError(f"action[1] must be {ACTION_ROLL} or {ACTION_STOP}, got {roll_action}")

    def check(self, action: object) -> tuple[bool, bool, str]:
        """Validate action and check if allowed given game state."""
        # Convert the numpy array to tuple if needed.
        if hasattr(action, "tolist"):
            action = tuple(action.tolist())  # pyright: ignore[reportUnknownVariableType, reportAttributeAccessIssue]

        # Check structure, must be a tuple of length 2.
        if not isinstance(action, tuple) or len(action) != ACTION_TUPLE_LENGTH:
            raise ValueError(f"action must be a tuple of length {ACTION_TUPLE_LENGTH}, got {type(action).__name__}")

        # Store validated action.
        self._action = cast("tuple[object,object]", action)

        # Validate each element
        self._validate_dice_index()
        self._validate_roll_action()

        # After validation cast to the correct type and heck game state.
        return self.is_allowed(cast("tuple[int,int]", self._action))

    def is_allowed(self, action: tuple[int, int]) -> tuple[bool, bool, str]:
        """Check if action is allowed."""
        self._terminated = False
        self._truncated = False

        # Selected Face value was not rolled.
        if self._dice.get_rolled()[action[ACTION_INDEX_DICE]] == 0:
            self._truncated = True
            self._explanation = RED + "Truncated: Selected Face value not rolled" + NO_RED
            return self._terminated, self._truncated, self._explanation

        # Dice already collected cannot be taken again.
        if self._dice.get_collected()[action[ACTION_INDEX_DICE]] != 0:
            self._truncated = True
            self._explanation = RED + "Truncated: Dice already collected cannot be taken again" + NO_RED
            return self._terminated, self._truncated, self._explanation

        remaining_dice = self._dice.get_rolled().copy()  # Copy in order not to overwrite the real rolled variable.
        remaining_dice[action[ACTION_INDEX_DICE]] = 0  # Overwrite with zero for the last face collected.

        # Try to roll when no dice left to roll.
        if action[ACTION_INDEX_ROLL] == ACTION_ROLL and not remaining_dice:
            self._truncated = True
            self._explanation = RED + "Truncated: No Dice left to roll and roll action selected." + NO_RED
            return self._terminated, self._truncated, self._explanation

        return self._terminated, self._truncated, self._explanation
