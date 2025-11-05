"""Class Checker."""

from pickomino_env.src.constants import (  # Coloured printouts, game and action constants.
    ACTION_INDEX_ROLL,
    ACTION_INDEX_DICE,
    ACTION_ROLL,
    ACTION_STOP,
    GREEN,
    NO_GREEN,
    NO_RED,
    NUM_DICE,
    RED,
    SMALLEST_TILE,
)
from pickomino_env.src.dice import Dice
from pickomino_env.src.player import Player
from pickomino_env.src.table_tiles import TableTiles


class Checker:
    """Class Checker."""

    def __init__(self, dice: Dice, players: list[Player], table_tiles: TableTiles):
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
            for rolled, collected in zip(self._dice.get_rolled(), self._dice.get_collected())
        )

        self._failed_attempt = not can_take
        self._explanation = (
            GREEN + "Good case" + NO_GREEN
            if can_take
            else RED + f"Failed: Collected was {self._dice.get_collected()}\n"
            f"No possible rolled dice to taken in {self._dice.get_rolled()}" + NO_RED
        )

        return self._failed_attempt, self._explanation

    def set_failed_no_tile_to_take(self, current_player_index: int, action: tuple[int, int]) -> tuple[bool, str]:
        """Failed: Not able to take a tile with dice sum reached."""
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

        # Check if no tile available on the table or from player to take.
        elif not self._table_tiles.find_next_lower_tile(self._dice.score()[0]) and steal_index is None:
            self._failed_attempt = True
            self._explanation = RED + "Failed: No tile on table or from another player can be taken" + NO_RED

        return self._failed_attempt, self._explanation

    def set_failed_no_worms(self, action: tuple[int, int]) -> tuple[bool, str]:
        """No worm collected and action stop."""
        if not self._dice.score()[1] and action[ACTION_INDEX_ROLL] == ACTION_STOP:
            self._failed_attempt = True
            self._explanation = RED + "Failed: No worm collected" + NO_RED

        return self._failed_attempt, self._explanation

    def action_is_allowed(self, action:tuple[int, int]) -> tuple[bool, bool, str]:
        """Check if action is allowed."""
        self._terminated = False
        self._truncated = False

        # Check action values are within range
        if action[ACTION_INDEX_DICE] not in range(0, 6) or action[ACTION_INDEX_ROLL] not in range(0, 2):
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

        remaining_dice = self._dice.get_rolled().copy() # Copy in order not to overwrite the real rolled variable.
        remaining_dice[action[ACTION_INDEX_DICE]] = 0 # Overwrite with zero for the one just collected.

        # Try to roll when no dice left to roll.
        if action[ACTION_INDEX_ROLL] == ACTION_ROLL and not remaining_dice:
            self._truncated = True
            self._explanation = RED + "Truncated: No Dice left to roll and roll action selected." + NO_RED
            return self._terminated, self._truncated, self._explanation

        return self._terminated, self._truncated, self._explanation
        # Get to here:Action allowed try to take a tile.


if __name__ == "__main__":
    test_dice = Dice()
    test_players: list[Player] = []
    test_tables_tiles = TableTiles()
    checker = Checker(test_dice, test_players, test_tables_tiles)
    checker.set_failed_already_collected()
