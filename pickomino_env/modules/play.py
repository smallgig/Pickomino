"""Play the game manually with pygame-ce rendering."""

from __future__ import annotations

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


if __name__ == "__main__":
    manual_play = ManualPlay()
    # Keep offering to play until the user does not want to play again.
    while True:  # pylint: disable=while-used
        game_number_of_bots: int = int(input("Enter number of bots you want to play against (0 - 6): "))
        game_env = PickominoEnv(game_number_of_bots, render_mode="human")
        manual_play.play(game_env)

        play_again: bool = bool(int(input("Play again? Enter '1', else '0': ")))
        if not play_again:
            break
