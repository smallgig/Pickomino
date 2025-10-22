"""Test bot."""

import numpy as np

from pickomino_env.src.bot import Bot

RED = "\033[31m"
NO_RED = "\033[0m"


class BotTest:
    def __init__(self) -> None:
        pass

    @staticmethod
    # If None is always the return value, it pops in console when playing a game.
    # The env should be type annotated env:PickominoEnv but leads to circular import.
    def play_manual(env, max_turns: int = 300) -> None:  # type: ignore [no-untyped-def]
        """Run interactive test."""
        observation, info = env.reset()
        reward: int = 0

        dice_rolled_coll = (
            observation["dice_collected"],
            observation["dice_rolled"],
        )

        print("Reset! Info before playing:")
        for key, value in info.items():
            print(key, value)

        for step in range(max_turns):
            print("Step:", step)
            print("Your showing tile: ", observation["tile_players"], "(your reward = ", reward, ")")
            print_roll(dice_rolled_coll, info["sum"], info["dice"])
            print("Tiles on table:", end=" ")

            for inde, tile in enumerate(observation["tiles_table"]):
                if tile:
                    print(inde + 21, end=" ")
            print()
            selection: int = int(input("Which dice do you want to collect? (1..5 or worm =6): ")) - 1
            stop: int = int(input("Keep rolling? (0 = ROLL,  1 = STOP): "))
            print()
            observation, reward, terminated, truncated, info = env.step((selection, stop))
            dice_rolled_coll = (
                observation["dice_collected"],
                observation["dice_rolled"],
            )
            print(f"Terminated: {terminated} Truncated:{truncated}")
            print(f'Explanation: {info["explanation"]}')
            print("Rolled: ", observation["dice_rolled"])
            print("Last returned tile:", info["last_returned_tile"])

            if terminated:
                observation, info = env.reset()

        return None

    @staticmethod
    def play_automated(env, max_turns: int = 1000):

        game_observation, game_info = env.reset()
        game_reward: int = 0
        game_total: object = 0
        game_terminated: bool = False
        game_truncated: bool = False

        bot = Bot()
        values = np.array([1, 2, 3, 4, 5, 5], int)

        dice_coll_rolled = game_observation["dice_collected"], game_observation["dice_rolled"]
        print("Reset")
        total_reward: int = 0
        step: int = 0
        for step in range(max_turns):
            print()
            print("==================================================================")
            print("Bot test running with Step:", step)
            print(
                "Your top showing tile: ",
                game_observation["tile_players"],
                "(Your latest reward = ",
                (RED + f"{game_reward}" + NO_RED) if game_reward < 0 else game_reward,
                ")",
            )
            print_roll(dice_coll_rolled, game_total, game_info["dice"])
            print("Tiles on table:", end=" ")
            for ind, game_tile in enumerate(game_observation["tiles_table"]):
                if game_tile:
                    print(ind + 21, end=" ")
                else:
                    print("_", end=" ")
            print()
            print("Explanation: ", (game_info["explanation"]))
            smallest_tile: int = int(str(game_info["smallest_tile"]))  # Hairy hack.
            selection, stop = bot.policy(
                game_observation["dice_rolled"],
                game_observation["dice_collected"],
                smallest_tile,
            )
            print("Action:")
            print(
                "     Selection (1-6):",
                selection + 1,  # Player starts with 1.
                "   (Sum after collecting = ",
                game_info["sum"] + game_observation["dice_rolled"][selection] * values[selection],
                ")",
            )
            print("     Finish?:", "Stop" if stop else "Roll")
            game_action = (selection, stop)
            game_observation, game_reward, game_terminated, game_truncated, game_info = env.step(game_action)
            total_reward += game_reward

            dice_coll_rolled = (
                game_observation["dice_collected"],
                game_observation["dice_rolled"],
            )
            game_total = game_info["sum"]
            failed_attempt = game_info["failed_attempt"]
            print(
                "Terminated:",
                game_terminated,
                "          Truncated:",
                game_truncated,
                "          Failed attempt:",
                failed_attempt,
            )
            print("Player Stack:", game_info["player_stack"])
            print("Last returned tile:", game_info["last_returned_tile"])
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
            game_observation["tile_players"],
            "(your reward = ",
            (RED + f"{game_reward}" + NO_RED) if game_reward < 0 else game_reward,
            ")",
        )
        print(f"Terminated: {game_terminated}")
        print(f"Truncated: {game_truncated}")
        print("Failed attempt:", game_info["failed_attempt"])


def print_roll(observation: tuple[list[int], list[int]], total: object, dice: object) -> None:
    """Print one roll."""
    print(dice)
    # Print line of collected dice.
    for collected in range(len(observation[0])):
        print(f"   {observation[0][collected]}      ", end="")
    # Print sum at the end of the collected dice line
    print(f" collected   (Sum = {total})")
    # Print line of rolled dice.
    for rolled in range(len(observation[1])):
        print(f"   {observation[1][rolled]}      ", end="")
    print(" rolled")
    print("----------------------------------------------------------")
