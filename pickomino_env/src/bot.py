from time import sleep

from numpy.ma.core import argmax

from pickomino_env import pickomino_gym_env
from pickomino_env.src.dice import Dice
from random import randint


class Bot:
    def __init__(self):
        pass

    def run(self):
        max_turns: int = 3000
        env = pickomino_gym_env.PickominoEnv(2)
        game_observation, game_info = env.reset()
        game_reward: int = 0
        game_total = 0
        game_terminated: bool = False
        dice_coll_rolled = game_observation["dice_collected"], game_observation["dice_rolled"]
        print("Reset")
        for step in range(max_turns):
            sleep(1)
            print("Step:", step)
            print("Your showing tile: ", game_observation["tile_players"], "(your reward = ", game_reward, ")")
            print_roll(dice_coll_rolled, game_total, game_info["dice"])
            print("Tiles on table:", end=" ")
            for ind, game_tile in enumerate(game_observation["tiles_table"]):
                if game_tile:
                    print(ind + 21, end=" ")
            print()
            # if self._dice.get_collected()[self._action[PickominoEnv.ACTION_INDEX_DICE]] != 0:
            #     if self._dice.get_rolled()[self._action[PickominoEnv.ACTION_INDEX_DICE]] != 0:
            #         self._terminated = True
            if game_terminated:
                print(game_info["terminated_reason"])

                selection: int = randint(0, 5)
            else:
                selection: int = argmax(dice_coll_rolled[1])
            stop: int = 0
            if game_total > 20:
                stop = 1

            print(selection, stop)
            print()
            game_action = (selection, stop)
            game_observation, game_reward, game_terminated, game_truncated, game_info = env.step(game_action)
            dice_coll_rolled = game_observation["dice_collected"], game_observation["dice_rolled"]
            game_total = game_info["sum"]
            print(game_terminated, game_truncated)


def print_roll(observation: tuple[list[int], list[int]], total: int, dice: object) -> None:
    """Print one roll."""
    print(dice)
    # Print line of collected dice.
    for collected in range(len(observation[0])):
        print(f"   {observation[0][collected]}    ", end="")
    # Print sum at the end of the collected dice line
    print(f" collected   sum = {total}")
    # Print line of rolled dice.
    for rolled in range(len(observation[1])):
        print(f"   {observation[1][rolled]}    ", end="")
    print(" rolled")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    bot = Bot()
    bot.run()
