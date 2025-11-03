from pickomino_env.src.dice import Dice
from pickomino_env.src.player import Player
from pickomino_env.src.table_tiles import TableTiles

RED = "\033[31m"
NO_RED = "\033[0m"
GREEN = "\033[32m"
NO_GREEN = "\033[0m"


class Checker:
    SMALLEST_TILE = 21
    LARGEST_TILE = 36
    ACTION_INDEX_DICE = 0
    ACTION_INDEX_ROLL = 1
    ACTION_ROLL = 0
    ACTION_STOP = 1
    NUM_DICE = 8

    def __init__(self):
        self._failed_attempt = False
        self._explanation = ""
        # Init dice object? or give it with method parameter


    def set_failed_already_collected(self, dice:Dice) -> tuple[bool, str]:
        """Check if a die is available to take."""
        can_take = any(
            rolled > 0 and collected == 0
            for rolled, collected in zip(dice.get_rolled(), dice.get_collected())
        )

        self._failed_attempt = not can_take
        self._explanation = (
            GREEN + "Good case" + NO_GREEN
            if can_take
            else RED + f"Failed: Collected was {dice.get_collected()}\n"
            f"No possible rolled dice to taken in {dice.get_rolled()}" + NO_RED
        )

        return self._failed_attempt, self._explanation

    def set_failed_no_tile_to_take(self, dice: Dice, players: list[Player], table_tiles:TableTiles, current_player_index:int, action:tuple[int, int]) -> tuple[bool, str]:
        """Failed: Not able to take a tile with dice sum reached."""
        # Environment takes the highest tile on the table or player stack.
        # Check if any tile can be picked from another player.
        # Index from player to steal.

        steal_index = next(
            (
                i
                for i, player in enumerate(players)
                if i != current_player_index and player.show() == dice.score()[0]
            ),
            None,
        )
        # pylint: disable=confusing-consecutive-elif
        if dice.score()[0] < self.SMALLEST_TILE:

            if action[self.ACTION_INDEX_ROLL] == self.ACTION_STOP:
                self._failed_attempt = True
                self._explanation = RED + "Failed: 21 not reached and action stop" + NO_RED

            if sum(dice.get_collected()) == self.NUM_DICE:
                self._failed_attempt = True
                self._explanation = RED + "Failed: 21 not reached and no dice left" + NO_RED

        # Check if no tile available on the table or from player to take.
        elif not table_tiles.find_next_lower_tile(dice.score()[0]) and steal_index is None:
            self._failed_attempt = True
            self._explanation = RED + "Failed: No tile on table or from another player can be taken" + NO_RED

        return self._failed_attempt, self._explanation

    def set_failed_no_worms(self, dice:Dice, action:tuple[int, int]) -> tuple[bool, str]:
        """No worm collected and action stop."""
        if not dice.score()[1] and action[self.ACTION_INDEX_ROLL] == self.ACTION_STOP:
            self._failed_attempt = True
            self._explanation = RED + "Failed: No worm collected" + NO_RED

        return self._failed_attempt, self._explanation