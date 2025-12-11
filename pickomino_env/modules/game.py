"""Class Game."""

# ruff: noqa: I001

from __future__ import annotations

import numpy as np

from pickomino_env.modules.constants import (
    ACTION_INDEX_DICE,
    ACTION_INDEX_ROLL,
    ACTION_ROLL,
    ACTION_STOP,
    GREEN,
    NO_GREEN,
    NO_RED,
    NUM_DICE,
    RED,
    LARGEST_TILE,
    SMALLEST_TILE,
)

__all__ = ["Game"]


class Game:  # pylint: disable=too-few-public-methods
    """Class Game."""

    class Dice:
        """Class Dice. Represents a collection of die face frequencies.

        An example for eight dice with six faces is: [0, 0, 3, 4, 0, 1]
        This example means that three threes, four fours and one worm die face have been collected.
        """

        def __init__(self, random_generator: np.random.Generator | None = None) -> None:
            """Initialize Dice."""
            self._random_generator = random_generator or np.random.default_rng()
            self.values: list[int] = [1, 2, 3, 4, 5, 5]  # Worm has value 5.
            self._n_dice: int = 8
            self._collected: list[int] = [
                0,
                0,
                0,
                0,
                0,
                0,
            ]  # Collected dice, up to 8 per side.
            self._rolled: list[int] = [0, 0, 0, 0, 0, 0]  # Last roll.

        def collect(self, action_face: int) -> list[int]:
            """Insert the chosen face to the collected list."""
            self._collected[action_face] = self._rolled[action_face]
            return self._collected

        def get_collected(self) -> list[int]:
            """Getter."""
            return self._collected

        def get_rolled(self) -> list[int]:
            """Getter."""
            return self._rolled

        def roll(self) -> list[int]:
            """Roll remaining dice."""
            self._rolled = [0, 0, 0, 0, 0, 0]

            for _ in range(self._n_dice - sum(self._collected)):
                self._rolled[self._random_generator.integers(0, 6)] += 1  # pyright:ignore[reportUnknownMemberType]

            return self._rolled

        def score(self) -> tuple[int, bool]:
            """Count the score based on an array of die face frequencies."""
            # Check if there is at least one worm
            has_worms = self._collected[-1] > 0
            # Multiply the frequency of each die face with its value
            current_score = int(np.dot(self.values, self._collected))  # pyright: ignore[reportUnknownMemberType]
            # Using the dice sum as an index in [21..36], higher rolls can only pick 36 or lower.
            current_score = int(min(current_score, LARGEST_TILE))
            return current_score, has_worms

        def __str__(self) -> str:
            """Print the dice."""
            die_faces: list[str] = [
                "",  # index = 0 doesn't have a face
                "[     ]\n[  @  ]\n[     ]",  # index 1
                "[@    ]\n[     ]\n[    @]",  # index 2
                "[@    ]\n[  @  ]\n[    @]",  # index 3
                "[@   @]\n[     ]\n[@   @]",  # index 4
                "[@   @]\n[  @  ]\n[@   @]",  # index 5
                "\033[31m[oo@@@]\033[0m\n\033[31m[WORM!]\033[0m\n\033[31m[@@@@@]\033[0m",  # index 6
            ]
            # Print one dice.
            show_values = [1, 2, 3, 4, 5, 6]
            faces = [die_faces[v].splitlines() for v in show_values]
            return_value = ""
            for line in zip(*faces):
                return_value += "   ".join(line) + "\n"
            return return_value

    class TableTiles:
        """Define the tiles on the table."""

        def __init__(self) -> None:
            """Construct the table tiles."""
            self._tile_table: dict[int, bool] = {
                21: True,
                22: True,
                23: True,
                24: True,
                25: True,
                26: True,
                27: True,
                28: True,
                29: True,
                30: True,
                31: True,
                32: True,
                33: True,
                34: True,
                35: True,
                36: True,
            }
            self.worm_values: list[int] = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]

        def set_tile(self, *, tile_number: int, is_available: bool) -> None:
            """Set one Tile."""
            self._tile_table[tile_number] = is_available

        def get_table(self) -> dict[int, bool]:
            """Get the tile table."""
            return self._tile_table

        def is_empty(self) -> bool:
            """Check if the table is empty."""
            return not self._tile_table.values()

        def highest(self) -> int:
            """Get the highest tile on the table."""
            highest = 0
            if not self.is_empty():
                for key, value in self._tile_table.items():
                    if value:
                        highest = key
            return highest

        def smallest(self) -> int:
            """Get the smallest tile on the table."""
            smallest = 0
            if not self.is_empty():
                for key, value in reversed(self._tile_table.items()):
                    if value:
                        smallest = key
            return smallest

        def find_next_lower_tile(self, dice_sum: int) -> int:
            """Find the next lower tile than the dice sum."""
            lowest = 0
            for key, value in self._tile_table.items():
                if key < dice_sum and value:
                    lowest = key
            return lowest

    class Player:
        """Player class with his tiles and name."""

        def __init__(self, *, bot: bool, name: str) -> None:
            """Initialize a player."""
            self.name: str = name
            self.tile_stack: list[int] = []
            self.bot: bool = bot

        def show(self) -> int:
            """Show the tile from the player stack."""
            if self.tile_stack:
                return self.tile_stack[-1]
            return 0

        def show_all(self) -> list[int]:
            """Show all tiles on the player stack."""
            if self.tile_stack:
                return self.tile_stack
            return [0]  # Zero indicates the stack is empty.

        def add_tile(self, tile: int) -> None:
            """Add a tile to the player stack."""
            self.tile_stack.append(tile)

        def remove_tile(self) -> int:
            """Remove the top tile from the player stack."""
            return self.tile_stack.pop()

        def end_score(self) -> int:
            """Return player score at the end of the game."""
            score: int = 0
            table = Game.TableTiles()
            for tile in self.tile_stack:
                score += table.worm_values[tile - SMALLEST_TILE]  # List of worm values count from zero.
            return score

    class Checker:
        """Class Checker."""

        def __init__(self, dice: Game.Dice, players: list[Game.Player], table_tiles: Game.TableTiles) -> None:
            """Initialize Checker."""
            self._failed_attempt = False
            self._terminated = False
            self._truncated = False
            self._explanation = ""
            self._dice = dice
            self._players = players
            self._table_tiles = table_tiles

        def set_failed_already_collected(self) -> tuple[bool, str]:
            """Check if a die is available to take."""
            can_take = any(
                rolled > 0 and collected == 0
                for rolled, collected in zip(
                    self._dice.get_rolled(),
                    self._dice.get_collected(),
                )
            )

            self._failed_attempt = not can_take
            self._explanation = (
                GREEN + "Good case" + NO_GREEN
                if can_take
                else RED + f"Failed: Collected was {self._dice.get_collected()}\n"
                f"No possible rolled dice to taken in {self._dice.get_rolled()}" + NO_RED
            )

            return self._failed_attempt, self._explanation

        def set_failed_no_tile_to_take(
            self,
            current_player_index: int,
            action: tuple[int, int],
        ) -> tuple[bool, str]:
            """Failed: Not able to take a tile with the dice sum reached."""
            # Environment takes the highest tile on the table or player stack.
            # Check if any tile can be picked from another player.
            # Index from player to steal.

            steal_index = next(
                (
                    i
                    for i, player in enumerate(self._players)
                    if i != current_player_index and player.show() == self._dice.score()[0]
                ),
                None,
            )
            # pylint: disable=confusing-consecutive-elif
            if self._dice.score()[0] < SMALLEST_TILE:
                if action[ACTION_INDEX_ROLL] == ACTION_STOP:
                    self._failed_attempt = True
                    self._explanation = RED + "Failed: 21 not reached and action stop" + NO_RED

                if sum(self._dice.get_collected()) == NUM_DICE:
                    self._failed_attempt = True
                    self._explanation = RED + "Failed: 21 not reached and no dice left" + NO_RED

            # Check if no tile available on the table or from a player to take.
            elif (
                not self._table_tiles.get_table()[self._dice.score()[0]]
                and not self._table_tiles.find_next_lower_tile(self._dice.score()[0])
                and steal_index is None
            ):
                self._failed_attempt = True
                self._explanation = RED + "Failed: No tile on table or from another player can be taken" + NO_RED

            return self._failed_attempt, self._explanation

        def set_failed_no_worms(self, action: tuple[int, int]) -> tuple[bool, str]:
            """Set failed attempt for no worm collected.

            No worm collected, but the action is to stop.
            """
            if not self._dice.score()[1] and action[ACTION_INDEX_ROLL] == ACTION_STOP:
                self._failed_attempt = True
                self._explanation = RED + "Failed: No worm collected" + NO_RED

            return self._failed_attempt, self._explanation

        def action_is_allowed(self, action: tuple[int, int]) -> tuple[bool, bool, str]:
            """Check if action is allowed."""
            self._terminated = False
            self._truncated = False

            # Check action values are within range
            if action[ACTION_INDEX_DICE] not in range(6) or action[ACTION_INDEX_ROLL] not in range(2):
                self._terminated = True
                self._explanation = RED + "Terminated: Action index not in range" + NO_RED
                return self._terminated, self._truncated, self._explanation

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
            # Get to here:Action allowed try to take a tile.

    def __init__(self) -> None:
        """Initialize Game."""
        self.dice: Game.Dice = Game.Dice()
        self.table_tiles: Game.TableTiles = Game.TableTiles()
        self.player: Game.Player = Game.Player(bot=False, name="You")
        self.players: list[Game.Player] = []
        self.checker = Game.Checker(self.dice, self.players, self.table_tiles)
