"""Pickomino game with gymnasium API."""

import random as rand
from typing import Optional
import numpy as np
import gymnasium as gym


class PickominoEnv(gym.Env):
    """The environment class."""

    def __init__(self) -> None:
        """Constructor."""
        self._action_index_dice: int = 0
        self._action_index_roll: int = 1
        self._action_roll: int = 0
        self._action_stop: int = 1
        self._num_dice: int = 8
        self._remaining_dice: int = self._num_dice
        self._terminated: bool = False
        self._truncated: bool = False
        # Define what the agent can observe.
        # Dict space gives us structured, human-readable observations.
        # 6 possible faces of the dice. Worm = index 0, Rest: index = faces value of die
        self._dice_collected: list[int] = [0, 0, 0, 0, 0, 0]  # Collected dice, up to 8 per side.
        self._dice_rolled: list[int] = [0, 0, 0, 0, 0, 0]  # Last roll.
        self._roll_counter: int = 0
        self._tile_table: dict[int, bool] = {
            21: True,
            22: True,
            23: True,
            24: True,
            25: True,
            26: True,
            27: True,
            28: True,
            29: True,
            30: True,
            31: True,
            32: True,
            33: True,
            34: True,
            35: True,
            36: True,
        }
        self.you: list[int] = []

        self.observation_space = gym.spaces.Dict(
            {
                "dice_collected": gym.spaces.Box(low=0, high=8, shape=(6,), dtype=np.int64),  # 6 * 8
                "dice_rolled": gym.spaces.Box(low=0, high=6, shape=(6,), dtype=np.int64),
                # "available_tiles": gym.spaces.Box(low=0, high=1, shape=(37,), dtype=np.int64),
                "tiles_table": gym.spaces.Dict(
                    {i: gym.spaces.Discrete(5) for i in range(21, 37)}  # Set the value to False/True when collected.
                ),
                "tiles_player": gym.spaces.Discrete(1),
            }
        )
        # Action space is a tuple. First action: which dice you take. Second action: roll again or not.
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(6), gym.spaces.Discrete(2)))

    def _get_dice_sum(self) -> int:
        """Return the sum of the collected dices."""
        return_value = 0

        # Dice with eyes = number of eyes per die times the number of dice.
        for ind, die in enumerate(self._dice_collected):
            return_value += ind * die  # Worms have ind = 0 and hence are not counted here.

        # Worms have value five.
        return_value += self._dice_collected[0] * 5

        return return_value

    def _get_obs_dice(self):  # -> tuple[list[int], list[int]] or npt arrays if we keep them
        """Convert internal state to observation format.

        Returns:
            tuple: Dices collected and dices rolled.
        """

        return self._dice_collected, self._dice_rolled

    def _get_obs_tiles(self):
        """Convert internal state to observation format.

        Returns:
            dict: Tiles distribution
        """
        observation_tiles = (self.you, self._tile_table)
        return observation_tiles

    def _get_info(self, action):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with additional information which is useful for debugging but not necessary for learning.
        """
        return_value = {
            "action": action,
            "self.num_dice": self._num_dice,
            "self.remaining_dice": self._remaining_dice,
            "self._dice_collected": self._dice_collected,
            "self._dice_rolled": self._dice_rolled,
            "self.roll_counter": self._roll_counter,
            "self.observation_space": self.observation_space,
            "self.action_space": self.action_space,
            "self._get_sum()": self._get_dice_sum(),
            "self._get_obs_dice()": self._get_obs_dice(),
            "self._get_obs_tiles()": self._get_obs_tiles(),
            "self.legal_move(action)": self._legal_move(),
        }
        return return_value

    def _soft_reset(self) -> None:
        self._dice_collected = [0, 0, 0, 0, 0, 0]
        self._dice_rolled = [0, 0, 0, 0, 0, 0]
        self._roll_counter = 0
        self._remaining_dice = 8
        print(f"PRINT DEBUGGING - rolling {self._num_dice} dice.")
        for _ in range(self._num_dice):
            self._dice_rolled[rand.randint(0, 5)] += 1

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        self._dice_collected: list[int] = [0, 0, 0, 0, 0, 0]
        self._dice_rolled: list[int] = [0, 0, 0, 0, 0, 0]
        self._roll_counter = 0
        self._remaining_dice = 8
        self._tile_table = {
            21: True,
            22: True,
            23: True,
            24: True,
            25: True,
            26: True,
            27: True,
            28: True,
            29: True,
            30: True,
            31: True,
            32: True,
            33: True,
            34: True,
            35: True,
            36: True,
        }
        self._terminated = False
        self._truncated = False

        print(f"PRINT DEBUGGING - rolling {self._num_dice} dice.")
        for _ in range(self._num_dice):
            self._dice_rolled[rand.randint(0, 5)] += 1

        return_obs = {
            "dice_collected": np.array(self._dice_collected),
            "dice_rolled": np.array(self._dice_rolled),
            # "player": gym.spaces.Discrete(num_players),  # Number of players in the game.
            "tiles_table": self._tile_table,  # Tiles that can be taken.
            "tiles_player": self.you,
        }
        # self._dice_rolled = np.array([0, 2, 1, 4, 1, 0])

        info = self._get_info((0, 1))  # Arbitrary action in reset only for debugging

        return return_obs, info

    def _legal_move(self):
        """Check if action is allowed."""
        self._terminated: bool = False
        self._truncated: bool = False

        # TODO prove illegal moves
        # Dice already collected cannot be taken again.
        if self._roll_counter >= 2:
            self._terminated = True
            for index in range(len(self._dice_rolled)):
                if self._dice_rolled[index] > 0 and self._dice_collected[index] == 0:
                    self._terminated = False

            # for die_collected in self._dice_collected:
            #     for die_rolled in self._dice_rolled:
            #         if die_rolled > 0 and die_collected == 0:
            #             self.terminated = False

        # No dice left and 21 not reached.
        if self._remaining_dice == 0 and self._get_dice_sum() < 21:
            self._terminated = True

        # No worm collected
        if self._remaining_dice == 0 and self._dice_collected[0] == 0:
            self._terminated = True

    def _step_dice(self, action: tuple[int, int]) -> None:
        """Execute one roll of the dice and picking or returning a tile.

        :param: action: The action to take: which dice to collect.
        """
        # Collect the dice according to the action.
        self._dice_collected[action[self._action_index_dice]] = self._dice_rolled[action[self._action_index_dice]]
        # TODO: Jarl: why do we need to do this? Does not make the list self._dice_rolled shorter??
        # TODO: we reset it in line 218
        self._dice_rolled.remove(self._dice_rolled[action[self._action_index_dice]])
        self._legal_move()  # Check before reset and after removing the collected dice
        # Reduce the remaining number of dice by the number collected.
        self._remaining_dice -= self._dice_collected[action[self._action_index_dice]]
        # Reset diced rolled before rolling again below.
        self._dice_rolled = [0, 0, 0, 0, 0, 0]

        # TODO: if terminated or truncated, we never roll again after resting diced rolled.
        # TODO: Suspect terminated or truncated is set wrongly some where!
        if self._terminated or self._truncated:
            return

        # Action is to roll
        if action[self._action_index_roll] == self._action_roll:
            # TODO: Jarl: we have updated self._remaining_dice, why do we need to do this?
            max_dice: int = self._num_dice - np.sum(self._dice_collected)
            dices_to_roll: int = min(self._remaining_dice, max_dice)
            print(f"PRINT DEBUGGING - rolling {dices_to_roll} dice.")
            for _ in range(dices_to_roll):
                self._dice_rolled[rand.randint(0, 5)] += 1
            # TODO: Jarl: recognise a misthrow here when rolled dices have all been collected already
            # TODO: and do somthing appropriate. Instead of asking the user to make an illegal move.
            self._roll_counter += 1
            self._truncated = False
        # Action is to stop rolling.
        else:
            self._truncated = True

    @staticmethod
    def get_worms(moved_key: int) -> int:
        """Give back the number of worms (1..4) for given the dice sum (21..36).
        Mapping:
        21–24 -> 1, 25–28 -> 2, 29–32 -> 3, 33–36 -> 4
        """

        if not 21 <= moved_key <= 36:
            raise ValueError("dice_sum must be between 21 and 36.")
        return (moved_key - 21) // 4 + 1

    def _step_tiles(self) -> int:
        """Pick or return a tile.

        Internal sub-step for picking or returning a tile after finishing rolling dice.

        :return: Value of moving the tile [-4 ... +4]
        """
        return_value = 0  # No tile is moved.
        dice_sum: int = self._get_dice_sum()
        print("PRINT DEBUGGING - dice_sum: ", dice_sum)

        # Environment takes the highest tile on the table.
        # Only pick a tile if it is on the table.
        if self._tile_table[dice_sum]:
            print("PRINT DEBUGGING - Picking tile:", dice_sum)
            self.you.append(dice_sum)  # Add the tile to the player.
            self._tile_table[dice_sum] = False  # Mark the tile as no longer on the table.
            return_value = self.get_worms(dice_sum)
            self._truncated = True
        # Tile is not available on the table
        else:
            # Pick the highest of the tiles smaller than the unavailable tile
            # Find the highest tile smaller than the dice sum.
            highest = 0
            for tile in range(21, dice_sum):
                if self._tile_table[tile]:
                    highest = tile
            if highest:  # Found the highest tile to pick from the table.
                print("PRINT DEBUGGING - Picking tile:", highest)
                self.you.append(highest)  # Add the tile to the player.
                self._tile_table[highest] = False  # Mark the tile as no longer on the table.
                return_value = self.get_worms(highest)
                self._truncated = True
            # Also no smaller tiles available -> have to return players showing tile if there is one.
            else:
                if self.you:
                    tile_to_return: int = self.you.pop()  # Remove the tile from the player.
                    print("PRINT DEBUGGING - Returning tile:", tile_to_return, "to the table.")
                    self._tile_table[tile_to_return] = True  # Return the tile to the table.
                    return_value = -self.get_worms(tile_to_return)  # Reward is MINUS the value of the returned tile.
                    # If the returned tile is not the highest, turn the highest tile around (set to False)
                    # Search for the highest tile to turn.
                    highest = 0
                    for tile in range(tile_to_return + 1, 37):
                        if self._tile_table[tile]:
                            highest = tile
                    # Turn the highest tile if there is one.
                    if highest:
                        self._tile_table[highest] = False
                        print("PRINT DEBUGGING - Turning tile:", highest, "on the table.")

            # TODO: remove old stuff
            # # Empty nothing happens, reward stays zero
            # if not self.you:
            #     pass
            # # You have at least one Tile and put it back to the Table
            # else:
            #     moved_key = self.you.pop()
            #     self._tile_table[moved_key] = True
            #     return_value = -self.get_worms(moved_key)
            #     # Moved Tile is highest
            #     if moved_key == max(self._tile_table):
            #         pass
            #     # Tile not available to taken for the rest of the Game
            #     else:
            #         self._tile_table[moved_key] = False
        print("PRINT DEBUGGING - Your tiles:", self.you)
        self._soft_reset()
        return return_value

    def step(self, action: tuple[int, int]):
        reward = 0
        self._legal_move()
        self._step_dice(action)
        if self._remaining_dice == 0 or action[self._action_index_roll] == self._action_stop:
            reward = self._step_tiles()

        if not self.you:
            self.you.append(0)

        return_obs = {
            "dice_collected": self._dice_collected,
            "dice_rolled": self._dice_rolled,
            # "player": gym.spaces.Discrete(num_players),  # Number of players in the game.
            "tiles_table": self._tile_table,  # Tiles that can be taken.
            "tiles_player": self.you[-1],
        }
        info = self._get_info(action)

        return return_obs, reward, self._terminated, self._truncated, info


# The next 18 lines, until 'print(*line)', were copied from Stack Overflow


die_faces: list[str] = [
    "",  # index = 0 doesn't have a face
    "[     ]\n[  0  ]\n[     ]",  # index 1
    "[0    ]\n[     ]\n[    0]",  # index 2
    "[0    ]\n[  0  ]\n[    0]",  # index 3
    "[0   0]\n[     ]\n[0   0]",  # index 4
    "[0   0]\n[  0  ]\n[0   0]",  # index 5
    "[0   0]\n[0   0]\n[0   0]",  # index 6
]


def print_dice(values: list[int]) -> None:
    """Print one dice."""
    faces = [die_faces[v].splitlines() for v in values]
    for line in zip(*faces):
        print(*line)


def print_roll(observation, total) -> None:
    """Print one roll."""
    print_dice([1, 2, 3, 4, 5, 6])
    # Print line of collected dice.
    for collected in range(len(observation[0]) - 1):
        print(f"   {observation[0][collected + 1]}    ", end="")
    print(f"   {observation[0][0]}    ", end="")
    # Print sum at the end of the collected dice line
    print(f" collected   sum = {total}")
    # Print line of rolled dice.
    for rolled in range(len(observation[1]) - 1):
        print(f"   {observation[1][rolled + 1]}    ", end="")
    print(f"   {observation[1][0]}    ", end="")
    print(" rolled")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    # Interactive test.
    # TODO: not yet used.
    # NUMBER_OF_DICE: int = 8
    # NUMBER_OF_PLAYERS: int = 2
    MAX_TURNS: int = 300
    env = PickominoEnv()
    observation, info = env.reset()
    reward: int = 0
    # for key, value in info.items():
    #     print(key, value)
    total: int = info["self._get_sum()"]
    dices_rolled_coll = observation["dice_collected"], observation["dice_rolled"]
    print("Reset")
    for step in range(MAX_TURNS):
        print("Step:", step)
        print("Your showing tile: ", observation["tiles_player"], "(your reward = ", reward, ")")
        print_roll(dices_rolled_coll, total)
        print("Tiles on table:", end=" ")
        for tile in observation["tiles_table"]:
            if observation["tiles_table"][tile]:
                print(tile, end=" ")
        print()
        selection: int = int(input("Which dice do you want to collect? (1..5 or worm =6): "))
        stop: int = int(input("Keep rolling? (0 = ROLL,  1 = STOP: "))
        print()
        if selection == 6:
            selection = 0  # Collecting a worm internally has index 0.
        action: tuple[int, int] = (selection, stop)
        observation, reward, terminated, truncated, info = env.step(action)
        dices_rolled_coll = observation["dice_collected"], observation["dice_rolled"]
        total = info["self._get_sum()"]
