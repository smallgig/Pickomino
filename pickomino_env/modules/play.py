"""Play the game manually with pygame-ce rendering."""

from __future__ import annotations

import argparse

from pickomino_env.modules.constants import MAX_BOTS
from pickomino_env.pickomino import PickominoEnv


class ManualPlay:  # pylint: disable=too-few-public-methods.
    """Manual play using the pygame-ce GUI."""

    MAX_TURNS: int = 300

    def __init__(self) -> None:
        """Initialize ManualPlay."""

    def play(self, env: PickominoEnv) -> None:
        """Play manual game."""
        env.reset()

        game_terminated: bool = False

        for _ in range(self.MAX_TURNS):
            if game_terminated:
                break

            # Wait for the mouse click
            action = None
            while action is None:  # pylint: disable=while-used # Polling loop: wait for mouse click.
                env.render()
                action = env.renderer.get_full_action()

            selection, roll_choice = action

            _, _, game_terminated, _, _ = env.step((selection, roll_choice))


def main() -> None:
    """Entry point for manual play."""
    parser = argparse.ArgumentParser(description="Play Pickomino manually.")
    parser.add_argument("--number-of-bots", type=int, default=1)
    args = parser.parse_args()

    if not 1 <= args.number_of_bots <= MAX_BOTS:
        parser.error(f"--number-of-bots must be between 1 and {MAX_BOTS}.")

    manual_play = ManualPlay()
    # Keep offering to play until the user does not want to play again.
    while True:  # pylint: disable=while-used
        game_env = PickominoEnv(args.number_of_bots, render_mode="human")
        manual_play.play(game_env)

        play_again: bool = bool(int(input("Play again? Enter '1', else '0': ")))
        if not play_again:
            break


if __name__ == "__main__":
    main()
