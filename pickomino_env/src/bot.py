from time import sleep

from numpy.ma.core import argmax
import numpy as np

from pickomino_env import pickomino_gym_env


class Bot:
    def __init__(self):
        self.roll_counter: int = 0

    def heuristic_policy(self, rolled, collected) -> tuple[int, int]:
        #     Heuristic Strategy:
        #     - On or after the third roll, take worms if you can.
        #     - Otherwise, take the die side that contributes the most points.
        #     - Quit as soon as you can take a tile."""
        action_roll = 0
        self.roll_counter += 1

        if sum(collected):
            self.roll_counter: int = 0
        values = [1, 2, 3, 4, 5, 5]

        # Set rolled[ind] to 0 if alreaddy collected
        for ind, die in enumerate(collected):
            if die:
                rolled[ind] = 0
        contribution = rolled * values
        action_dice = argmax(contribution)

        if self.roll_counter >= 3 and not collected[5] and rolled[5]:
            action_dice = 5

        # Quit as soon as you can take a tile.
        if sum(collected * values) + contribution[action_dice] > 20:
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
            sleep(0.1)
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
            print(game_info["explanation"])
            selection, stop = self.heuristic_policy(game_observation["dice_rolled"], game_observation["dice_collected"])

            print(selection, stop)
            print()
            game_action = (selection, stop)
            game_observation, game_reward, game_terminated, game_truncated, game_info = env.step(game_action)
            total_reward += game_reward

            dice_coll_rolled = game_observation["dice_collected"], game_observation["dice_rolled"]
            game_total = game_info["sum"]
            print(
                f"Terminated: {game_terminated}, Truncated: {game_truncated}, Failed attempt: {game_info['failed_attempt']}"
            )
            print(f"Player Stack: {game_info["player_stack"]}")
            print(total_reward)


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
    bot.run("heuristic")
