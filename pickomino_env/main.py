"""Test bot."""

from typing import cast

import numpy as np

from pickomino_env.pickomino_gym_env import PickominoEnv
from pickomino_env.src.bot import Bot
from pickomino_env.src.constants import (  # Coloured printouts, game and action constants.
    NO_RED,
    RED,
    SMALLEST_TILE,
)
from pickomino_env.src.dice import Dice


class BotTest:
    """Test using bots."""

    values = np.array([1, 2, 3, 4, 5, 5], int)
    MAX_TURNS: int = 300

    def __init__(self) -> None:
        pass

    def play_manual_game(self, env: PickominoEnv) -> None:
        """Run interactive test."""
        game_observation, game_info = env.reset()
        game_reward: int = 0

        print("Reset! Info before playing:")
        for key, value in game_info.items():
            print(key, value)

        game_terminated: bool = False

        for step in range(self.MAX_TURNS):
            if game_terminated:
                break

            print("Step:", step)
            print("Your showing tile: ", game_observation["tile_players"], "(your reward = ", game_reward, ")")
            print_roll(
                game_observation["dice_collected"],
                game_observation["dice_rolled"],
                cast(Dice, game_info["dice"]).score()[0],
                game_info["dice"],
            )
            print("Tiles on table:", end=" ")

            for ind, game_tile in enumerate(game_observation["tiles_table"]):
                if game_tile:
                    print(ind + SMALLEST_TILE, end=" ")
            print()
            selection: int = int(input("Which dice do you want to collect? (1..5 or worm =6): ")) - 1
            stop: int = int(input("Keep rolling? (0 = ROLL,  1 = STOP): "))
            print()
            game_observation, game_reward, game_terminated, game_truncated, game_info = env.step((selection, stop))

            print(f"Terminated: {game_terminated} Truncated:{game_truncated}")
            print(f'Explanation: {game_info["explanation"]}')
            print("Rolled: ", game_observation["dice_rolled"])
            # print("Last returned tile:", game_info["last_returned_tile"])

    def play_automated(self, env: PickominoEnv) -> None:
        """Play automated game."""
        game_obs, game_info = env.reset()
        game_reward: int = 0
        game_total: object = 0
        game_terminated: bool = False
        game_truncated: bool = False

        bot = Bot()

        print("Reset")
        total_reward: int = 0
        step: int = 0
        for step in range(self.MAX_TURNS):
            print()
            print("==================================================================")
            print("Bot test running with Step:", step)
            print(
                "Your top showing tile: ",
                game_obs["tile_players"],
                "(Your latest reward = ",
                (RED + f"{game_reward}" + NO_RED) if game_reward < 0 else game_reward,
                ")",
            )
            print_roll(game_obs["dice_collected"], game_obs["dice_rolled"], game_total, game_info["dice"])
            print("Tiles on table:", end=" ")
            for ind, game_tile in enumerate(game_obs["tiles_table"]):
                if game_tile:
                    print(ind + SMALLEST_TILE, end=" ")
                else:
                    print("_", end=" ")
            print()
            print("Explanation: ", (game_info["explanation"]))
            selection, stop = bot.policy(
                game_obs["dice_rolled"],
                game_obs["dice_collected"],
                int(str(game_info["smallest_tile"])),  # Hairy hack.
            )
            print("Action:")
            print(
                "     Selection (1-6):",
                selection + 1,  # Player starts with 1.
                "   (Sum after collecting = ",
                cast(Dice, game_info["dice"]).score()[0] + game_obs["dice_rolled"][selection] * self.values[selection],
                ")",
            )
            print("     Finish?:", "Stop" if stop else "Roll")
            game_obs, game_reward, game_terminated, game_truncated, game_info = env.step((selection, stop))
            total_reward += game_reward
            game_total = cast(Dice, game_info["dice"]).score()[0]
            print("Terminated:", game_terminated, "          Truncated:", game_truncated)
            print("Player Stack:", game_info["player_stack"])
            # print("Last returned tile:", game_info["last_returned_tile"])
            print("Total reward:", total_reward)
            print()
            if game_terminated:
                break
        print()
        print()
        print("===================================================================")
        print("===================================================================")
        print("Final state:")
        print("Step:", step)
        print("Explanation:", game_info["explanation"])
        print("Player Stack:", game_info["player_stack"])
        print("Total reward (Score):")
        print(RED + f"{total_reward}" + NO_RED)
        print(
            "Your showing tile: ",
            game_obs["tile_players"],
            "(your reward = ",
            (RED + f"{game_reward}" + NO_RED) if game_reward < 0 else game_reward,
            ")",
        )
        print(f"Terminated: {game_terminated}")
        print(f"Truncated: {game_truncated}")


def print_roll(collected: list[int], rolled: list[int], total: object, dice: object) -> None:
    """Print one roll."""
    print(dice)
    # Print line of collected dice.
    for die in collected:
        print(f"   {die}      ", end="")
    # Print sum at the end of the collected dice line
    print(f" collected   (Sum = {total})")
    # Print line of rolled dice.
    for die in rolled:
        print(f"   {die}      ", end="")
    print(" rolled")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    # Keep offering to play until user does not want to play again.
    while True:  # pylint: disable=while-used
        game_bot = BotTest()
        mode_auto: bool = bool(int(input("For automatic play enter '1', for manual enter '0': ")))
        game_number_of_bots: int = int(input("Enter number of bots you want to play against (0 - 6): "))
        game_env = PickominoEnv(game_number_of_bots)
        if mode_auto:
            game_bot.play_automated(game_env)
        else:
            game_bot.play_manual_game(game_env)
        print()
        print(f"{RED} ============================================= {NO_RED}")
        print(f"{RED}Game ended! Printing results is still open.{NO_RED}")
        play_again: bool = bool(int(input("Play again? Enter '1', else '0': ")))
        print("You entered: ", play_again)
        if not play_again:
            break
