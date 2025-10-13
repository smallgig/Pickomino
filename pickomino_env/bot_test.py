"""Test bot"""

import numpy as np
from pickomino_env.src.bot import Bot
from pickomino_env.pickomino_gym_env import PickominoEnv

RED = "\033[31m"
NO_RED = "\033[0m"


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


max_turns: int = 100000
env = PickominoEnv(0)
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
    selection, stop = bot.heuristic_policy(
        game_observation["dice_rolled"], game_observation["dice_collected"], game_info["smallest_tile"]
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

    dice_coll_rolled = game_observation["dice_collected"], game_observation["dice_rolled"]
    game_total = game_info["sum"]
    failed_attempt = game_info["failed_attempt"]
    print(
        "Terminated:",
        game_terminated,
        "          Truncated:",
        game_truncated,
        "          Failed attempt:",
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
print(f"Player Stack: {game_info['player_stack']}")
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
print(f"Failed attempt: {game_info['failed_attempt']}")
