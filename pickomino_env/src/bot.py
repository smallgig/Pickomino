from time import sleep

import numpy as np
from numpy.ma.core import argmax
from pickomino_env import pickomino_gym_env

RED = "\033[31m"
NO_RED = "\033[0m"




class Bot:
    def __init__(self):
        self.roll_counter: int = 0

    def heuristic_policy(self, rolled, collected, smallest) -> tuple[int, int]:
        #     Heuristic Strategy:
        #     - On or after the third roll, take worms if you can.
        #     - Otherwise, take the die side that contributes the most points.
        #     - Quit as soon as you can take a tile."""
        action_roll = 0
        self.roll_counter += 1
        rolled = np.array(rolled)
        values = np.array([1, 2, 3, 4, 5, 5], int)

        if sum(collected):
            self.roll_counter: int = 0


        # Set rolled[ind] to 0 if already collected
        for ind, die in enumerate(collected):
            if die:
                rolled[ind] = 0
        contribution = rolled * values
        action_dice = argmax(contribution)

        if self.roll_counter >= 3 and not collected[5] and rolled[5]:
            action_dice = 5

        # Quit as soon as you can take a tile.
        if sum(collected * values) + contribution[action_dice] >= smallest:
            action_roll = 1

        return action_dice, action_roll

    def run(self, policy: str):
        max_turns: int = 300000
        env = pickomino_gym_env.PickominoEnv(2)
        game_observation, game_info = env.reset()
        game_reward: int = 0
        game_total = 0
        game_terminated: bool = False
        game_truncated: bool = False
        dice_coll_rolled = game_observation["dice_collected"], game_observation["dice_rolled"]
        print("Reset")
        total_reward: int = 0
        for step in range(max_turns):
            print()
            print("==================================================================")
            print("Step:", step)
            print(
                "Your showing tile: ",
                game_observation["tile_players"],
                "(your reward = ",
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
            selection, stop = self.heuristic_policy(
                game_observation["dice_rolled"], game_observation["dice_collected"], game_info["smallest_tile"]
            )
            print("Action:")
            print(
                "     Selection (1-6):",
                selection + 1,
                f"   (Sum after collecting = {game_info['sum'] + \
                game_observation['dice_rolled'][selection] * Bot.values[selection]})",
            )
            print("     Finish?:", "Stop" if stop else "Roll")
            game_action = (selection, stop)
            game_observation, game_reward, game_terminated, game_truncated, game_info = env.step(game_action)
            total_reward += game_reward

            dice_coll_rolled = game_observation["dice_collected"], game_observation["dice_rolled"]
            game_total = game_info["sum"]
            failed_attempt: bool = game_info["failed_attempt"]
            print(
                f"Terminated:",
                game_terminated,
                f"          Truncated:",
                game_truncated,
                f"          Failed attempt:",
                failed_attempt,  # TODO: consider removing as it is ALWAYS False!
            )
            print(f"Player Stack: {game_info['player_stack']}")
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
        print(f"Player Stack: {game_info["player_stack"]}")
        print("Total reward (Score):", RED + f"{total_reward}" + NO_RED)
        print(
            "Your showing tile: ",
            game_observation["tile_players"],
            "(your reward = ",
            (RED + f"{game_reward}" + NO_RED) if game_reward < 0 else game_reward,
            ")",
        )
        print(f"Terminated: {game_terminated}")
        print(f"Truncated: {game_truncated}")
        print(f"Failed attempt: {game_info['failed_attempt']}")


def print_roll(observation: tuple[list[int], list[int]], total: int, dice: object) -> None:
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


if __name__ == "__main__":
    bot = Bot()
    bot.run("heuristic")
