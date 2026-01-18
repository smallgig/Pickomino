"""Pickomino dice and tile collection game with Gymnasium API.

Based on the board game Pickomino by Reiner Knizia.
Gymnasium: https://gymnasium.farama.org/
Pickomino game rules: https://github.com/smallgig/Pickomino/blob/main/pickomino-rulebook.pdf
"""

from __future__ import annotations

__all__ = ["PickominoEnv"]

import time
from typing import Any

import gymnasium as gym
import numpy as np

from pickomino_env.modules.action_checker import ActionChecker
from pickomino_env.modules.bot import Bot
from pickomino_env.modules.constants import (
    ACTION_INDEX_DICE,
    ACTION_INDEX_ROLL,
    ACTION_ROLL,
    ACTION_STOP,
    LARGEST_TILE,
    MAX_BOTS,
    NUM_DICE,
    RENDER_DELAY,
    SMALLEST_TILE,
)
from pickomino_env.modules.dice import Dice
from pickomino_env.modules.game import Game
from pickomino_env.modules.player import Player
from pickomino_env.modules.renderer import Renderer
from pickomino_env.modules.rule_checker import RuleChecker


class PickominoEnv(gym.Env):  # type: ignore[type-arg]
    """Pickomino dice and tile collection game environment.

    ## Description

    Pickomino is a push-your-luck dice game where players compete to collect numbered tiles (21-36) by rolling dice.
    On each turn, a player rolls 8 dice and collects one die face at a time (without duplicates). The sum of the
    collected dice determines, which tile can be collected. Players then choose to either stop and take the tile or
    roll the remaining dice again to reach a higher-value tile.

    Rolling only die faces already collected causes a bust. The player must return their top tile to the table. The
    game ends when no tiles remain on the table. The winner has the most worm symbols across all their claimed tiles.

    ## Action Space

    A tuple `(dice_face, roll_choice)` of two discrete values:

    **dice_face** (0-5): Which die face to collect from the current roll
        - 0: Collect all 1's
        - 1: Collect all 2's
        - 2: Collect all 3's
        - 3: Collect all 4's
        - 4: Collect all 5's
        - 5: Collect all worm symbols
    **roll_choice** (0-1): Decision after collecting dice
        - 0: Roll the remaining dice again.
        - 1: Stop and claim a tile (end turn).

    ## Observation Space

    A structured dictionary with 4 keys:

    | Key | Shape | Range | Type | Description |
    |-----|-------|-------|------|-------------|
    | `dice_collected` | (6,) | [0, 8] | int64 | Count of each die face collected this turn. |
    | `dice_rolled` | (6,) | [0, 8] | int64 | Count of each die face in the current roll. |
    | `tiles_table` | (16,) | {0, 1} | int8 | Binary: which tiles (21-36) are available on the table. |
    | `tile_players` | (num_players,) | [0, 36] | int8 | Current tile held by each player (0=none). |.

    The tiles have values from 21 to 36, but the observation space must be continuous, and value zero indicates no
    tile. Hence, the range [0, 36].

    ## Rewards

    Reward is the worm value of the tile claimed or returned during the turn:

    **+4 to +1**: Successfully claiming a high-value tile (36, 35, ..., 21)
    **-1 to -4**: Returning a tile due to bust (A tile being stolen gives no (zero) reward)


    Each tile (21-36) carries 1-4 worm symbols. The reward matches the worm count.

    ## Starting State

    Each episode (full game) initializes with:
    * All 16 tiles (21-36) visible and available on the table
    * No players holding tiles
    * 8 dice rolled and ready
    * Agent (player 0) taking the first turn.

    ## Episode End

    **Terminated**: No tiles remain on the table.
    The game cannot continue.

    **Truncated**: Agent attempts an invalid action:
        * Collecting a die face not in the current roll
        * Collecting a duplicate die face
        * Taking an impossible action given the game state.

    ## Arguments

    env = gymnasium.make("Pickomino-v0”, render_mode="human")
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step((0, 1))

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `number_of_bots` | int | -- | the number of bot opponents (1-6). |
    | `render_mode` | str or None | None | Rendering: None, “human”, or "rgb_array". |.

    Raises:
        ValueError: If `number_of_bots` is not in [1, 6] or invalid `render_mode`.

    """

    def __init__(self, number_of_bots: int, render_mode: str | None = None) -> None:
        """Initialize a Pickomino game environment.

        Creates a new Pickomino environment with the specified number of bot opponents and rendering mode. The human
        player is always player 0. Bots are players 1+.

        Args:
            number_of_bots: Number of bot opponents (1-6).
            render_mode: How to render the game. Choices:
                - None: No rendering (default)
                - "human": Display game to screen in real-time
                - "rgb_array": Return an RGB array for recording.

        Raises:
            ValueError: If number_of_bots not in [1, MAX_BOTS] or invalid render_mode.

        """
        # Check inputs.
        if number_of_bots < 1 or number_of_bots > MAX_BOTS:
            raise ValueError(f"number_of_bots must be between 1 and {MAX_BOTS}, got {number_of_bots}.")
        valid_modes: set[str | None] = {None, "human", "rgb_array"}
        if render_mode not in valid_modes:
            raise ValueError(f"render_mode must be on of {valid_modes}, got '{render_mode}'")

        self._action: tuple[int, int] = 0, 0
        self._number_of_bots: int = number_of_bots
        self._game: Game = Game()
        self._create_players()
        # Define what the AI Agent can observe.
        # Dict space gives us structured, human-readable observations.
        # 6 possible faces of the dice. Max 8 dice.
        self.observation_space = gym.spaces.Dict(
            {
                "dice_collected": gym.spaces.Box(
                    low=0,
                    high=NUM_DICE,
                    shape=(6,),
                    dtype=np.int64,
                ),
                "dice_rolled": gym.spaces.Box(
                    low=0,
                    high=NUM_DICE,
                    shape=(6,),
                    dtype=np.int64,
                ),
                # Flatten the tiles into a 16-length binary vector. Needed for SB3 compatibility.
                # Nested dicts are not supported by SB3.
                "tiles_table": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(16,),
                    dtype=np.int8,
                ),
                "tile_players": gym.spaces.Box(
                    low=0,
                    high=LARGEST_TILE,
                    shape=(len(self._game.players),),
                    dtype=np.int8,
                ),
            },
        )
        # Action space is a tuple. First action: which dice to take. Second action: roll again or not.
        self.action_space = gym.spaces.MultiDiscrete([6, 2])
        self._render_mode = render_mode
        self._renderer = Renderer(self._render_mode)

    def render(self) -> np.ndarray | list[np.ndarray] | None:  # type: ignore[override]
        """Render the current game state.

            Renders the environment to screen or returns an RGB array depending on render_mode.

        Returns:
            If render_mode='rgb_array': RGB array of shape (height, width, 3).
            If render_mode='human': None (displays to pygame window).

        Raises:
            ValueError: If render_mode was None at initialization.

        """
        if self._render_mode is None:
            raise ValueError(
                "render() called with render_mode=None."
                "Specify render_mode when creating environment: "
                "e.g. gymnasium.make('Pickomino-v0', render_mode='human')",
            )
        return self._renderer.render(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            self._game.dice,
            self._game.players,
            self._game.tiles,
            self._game.current_player_index,
        )

    def _create_players(self) -> None:
        """Create the human (agent) player (player 0) and bot opponents with assigned names."""
        names = ["Alfa", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
        self._game.players.append(self._game.you)
        for i in range(self._number_of_bots):
            self._game.players.append(Player(bot=True, name=names[i]))

    def _tiles_vector(self) -> np.ndarray[Any, np.dtype[Any]]:
        """Return available tiles as a flat binary vector of length 16 for indexes 21..36."""
        return np.array(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            [1 if self._game.tiles.get_tiles()[i] else 0 for i in range(SMALLEST_TILE, LARGEST_TILE + 1)],
            dtype=np.int8,
        )

    def _current_obs(self) -> dict[str, object]:
        """Return the current observation."""
        return {
            "dice_collected": np.array(self._game.dice.get_collected()),  # pyright: ignore[reportUnknownMemberType]
            "dice_rolled": np.array(self._game.dice.get_rolled()),  # pyright: ignore[reportUnknownMemberType]
            "tiles_table": self._tiles_vector(),
            "tile_players": np.array(  # pyright: ignore[reportUnknownMemberType]
                [p.show() for p in self._game.players],
                dtype=np.int8,
            ),
        }

    def _get_info(self) -> dict[str, object]:
        """Return auxiliary information useful for debugging but not for learning."""
        return {
            "dice_collected": list(self._game.dice.get_collected()),
            "dice_rolled": list(self._game.dice.get_rolled()),
            "terminated": self._game.terminated,
            "tiles_table_vec": self._tiles_vector(),
            "smallest_tile": self._game.tiles.smallest(),
            "explanation": self._game.explanation,
            "player_stack": self._game.players[0].show_all(),
            "player_score": self._game.players[0].end_score(),
            "bot_scores": [player.end_score() for player in self._game.players[1:]],
        }

    def _end_of_turn_reset(self) -> None:
        """Reset all dice and checkers. Reset failed attempt. Roll again."""
        self._game.dice = Dice()
        self._game.rule_checker = RuleChecker(self._game.dice, self._game.players, self._game.tiles)
        self._game.action_checker = ActionChecker(self._game.dice)
        self._game.failed_attempt = False
        self._game.dice.roll()

    def _remove_tile_from_player(self) -> int:
        """Remove and return the current player's top tile to the table.

        Called when a player fails (for example, duplicate die face) or chooses to stop without reaching a valid tile.
        The tile is returned to the table, and the highest available tile is turned face-down (removed from play).

        Returns:
            Negative worm value of returned tile, or 0 if a player has no tile.
        """
        return_value = 0
        if self._game.players[self._game.current_player_index].show():
            tile_to_return: int = self._game.players[
                self._game.current_player_index
            ].remove_tile()  # Remove the tile from the player.
            self._game.tiles.get_tiles()[tile_to_return] = True  # Return the tile to the table.
            worm_index = tile_to_return - SMALLEST_TILE
            return_value = -self._game.tiles.worm_values[worm_index]  # Reward is MINUS the value of the worm value.
            # If the returned tile is not the highest, turn the highest tile face down by setting it to False.
            # Search for the highest tile to turn.
            highest = self._game.tiles.highest()
            # Turn the highest tile if there is one.
            if highest:
                self._game.tiles.set_tile(tile_number=highest, is_available=False)
        return return_value

    # noqa: RUF100, ARG002 external API constraint.
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new episode.

        Reset the environment to an initial state. Includes reinitializing the game,
        players and rolling the initial dice.

        Args:
            seed: Random seed for reproducible episodes. If None, the environment's Pseudo-Random Number Generator
            (PRNG) is not reset.
            options: Additional configuration options, unused.

        Returns:
            observation: Initial observation (dice_collected, dice_rolled, tiles_table, tile_players)
            info: Additional information meant for debugging.

        Raises:
            ValueError: If seed is not an int or None, or if options is not a dict or None.

        """
        # Check inputs.
        # Runtime validation: user-facing API boundary, enforce types despite annotations.
        if seed is not None and (not isinstance(seed, int) or seed < 0):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"seed must be a non-negative integer or None, got {seed}")
        if options is not None and not isinstance(options, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"options must be a dict or None, got {type(options).__name__}")

        # IMPORTANT. Must call this first. Seed the random number generator.
        super().reset(seed=seed)
        self._game = Game(random_generator=self.np_random)
        self._create_players()
        self._game.dice.roll()
        return self._current_obs(), self._get_info()

    def _step_dice(self) -> None:
        """Collect the chosen die face and optionally roll remaining dice.

        Collects all dice showing the chosen die face (from self._action).
        Validates the collection against game rules (no duplicates, tile available,
        worms present). If the action is to roll again, rolls remaining dice and validates
        again. Sets self._game.failed_attempt and explanation on rule violations.
        """
        # Check rules before collecting dice.
        self._game.failed_attempt, self._game.explanation = self._game.rule_checker.set_failed_already_collected()

        self._game.dice.collect(self._action[ACTION_INDEX_DICE])

        # Check rules again after collecting dice.
        self._game.failed_attempt, self._game.explanation = self._game.rule_checker.set_failed_no_tile_to_take(
            self._game.current_player_index,
            self._action,
        )
        self._game.failed_attempt, self._game.explanation = self._game.rule_checker.set_failed_no_worms(
            self._action,
        )

        # Action is to roll
        if self._action[ACTION_INDEX_ROLL] == ACTION_ROLL:
            self._game.dice.roll()
            self._game.failed_attempt, self._game.explanation = self._game.rule_checker.set_failed_already_collected()
            self._game.failed_attempt, self._game.explanation = self._game.rule_checker.set_failed_no_tile_to_take(
                self._game.current_player_index,
                self._action,
            )
            self._game.failed_attempt, self._game.explanation = self._game.rule_checker.set_failed_no_worms(
                self._action,
            )

    def _steal_from_bot(self, steal_index: int) -> int:
        """Steal an opponent's tile and award it to the current player.

        Args:
            steal_index: Index of opponent to steal from.

        Returns:
            Worm value of stolen tile.
        """
        tile_to_return: int = self._game.players[steal_index].remove_tile()  # Remove the tile from the player.
        self._game.players[self._game.current_player_index].add_tile(tile_to_return)
        worm_index = tile_to_return - SMALLEST_TILE
        return self._game.tiles.worm_values[worm_index]

    def _step_tiles(self) -> int:
        """Pick or return a tile based on the dice sum at the end of the turn.

        If turn failed: return a player's top tile to the table.
        Otherwise: steal from an opponent or take from the table.
        Handles fallback to lower-value tiles if the exact sum is unavailable.

        Returns:
            Reward: worm value of tile moved (positive if claimed, negative if returned).
        """
        dice_sum: int = self._game.dice.score()[0]

        # Tile selection priority:
        # 1. If failed: return own tile.
        # 2. If an opponent holds a matching tile: steal it.
        # 3. If an exact match on the table: claim it.
        # 4. If unavailable: claim the next lower tile.
        # 5. If no lower tiles exist: return the player's own tile.

        # Using the dice sum as an index in [21..36] below. Hence, for dice_sum < 21 need to return early.
        # A failed attempt or 21 was not reached. Return the tile to the table.
        if self._game.failed_attempt:
            return_value = self._remove_tile_from_player()
            self._end_of_turn_reset()
            return return_value
        # Takes the highest possible tile on the table, given the sum.
        # Check if any tile can be stolen from another player.
        # Index from player to steal.
        steal_index = next(
            (
                i
                for i, player in enumerate(self._game.players)
                if i != self._game.current_player_index and player.show() == dice_sum
            ),
            None,
        )
        if steal_index is not None:
            return_value = self._steal_from_bot(steal_index)

        # Exact tile available on the table.
        elif self._game.tiles.get_tiles()[dice_sum]:
            self._game.players[self._game.current_player_index].add_tile(
                dice_sum,
            )  # Add the tile to the player or bot.
            self._game.tiles.set_tile(
                tile_number=dice_sum,
                is_available=False,
            )  # Mark the tile as no longer on the table.
            worm_index = dice_sum - SMALLEST_TILE
            return_value = self._game.tiles.worm_values[worm_index]
        # Exact tile unavailable: fallback to the highest tile lower than the dice sum.
        else:
            # Pick the highest of the tiles smaller than the unavailable tile.
            highest: int = self._game.tiles.find_next_lower(dice_sum)
            if highest:  # Found the highest tile to pick from the table.
                self._game.players[self._game.current_player_index].add_tile(
                    highest,
                )  # Add the tile to the player.
                self._game.tiles.set_tile(
                    tile_number=highest,
                    is_available=False,
                )  # Mark the tile as no longer on the table.
                worm_index = highest - SMALLEST_TILE
                return_value = self._game.tiles.worm_values[worm_index]
            # No smaller tiles are available -> have to return the player's own top tile, if there is one.
            else:
                return_value = self._remove_tile_from_player()
                self._game.explanation = "No available tile on the table to take"

        self._end_of_turn_reset()
        return return_value

    def _play_bot(self) -> None:
        """Execute the turns for all bot players sequentially.

        Iterates through each bot opponent (players 1 through num_bots) and calls
        _step_bot() to execute their turn using the fixed bot heuristic. Each bot
        makes decisions based on current dice and board state until their turn ends.
        """
        bot = Bot()
        bot_action: tuple[int, int] = 0, 0

        # Player 0 is you, from 1 and onwards are the bots.
        for player in self._game.players[1:]:
            if player.bot:
                # The bot keeps rolling (ACTION ROLL) until
                # the heuristic decides to stop or, the game is over or, it is a failed attempt.
                # pylint: disable=while-used
                while (
                    bot_action[ACTION_INDEX_ROLL] == ACTION_ROLL  # Want to roll.
                    and not self._game.terminated  # Not game over.
                    and not self._game.failed_attempt  # Not a failed attempt.
                ):
                    bot_action = bot.policy(
                        self._game.dice.get_rolled(),
                        self._game.dice.get_collected(),
                        self._game.tiles.smallest(),
                    )
                    self._step_bot(bot_action)
                if self._render_mode is not None:
                    self.render()  # pyright: ignore[reportUnknownMemberType]
                    time.sleep(RENDER_DELAY)
            bot_action = 0, 0  # Reset for next player.
            self._game.current_player_index += 1

    def _step_bot(self, action: tuple[int, int]) -> None:
        """Execute a single turn for the current bot player.

        The bot collects a die face from the current roll and decides whether to
        roll again or stop and claim a tile. Updates the game state, handles tile
        acquisition or loss, and rolls new dice for the next player's turn.

        Args:
            action: dice_index, roll_choice representing the bot's decision.
        """
        self._action = action
        self._game.terminated, self._game.truncated, self._game.explanation = self._game.action_checker.is_allowed(
            action,
        )

        # Stop immediately if action was not allowed or similar.
        if self._game.terminated or self._game.truncated:
            self._end_of_turn_reset()
            return

        # Collect and roll the dice.
        self._step_dice()

        # Action is stop, or it is a failed attempt -> pick (or return) tile.
        if self._action[ACTION_INDEX_ROLL] == ACTION_STOP or self._game.failed_attempt:
            self._step_tiles()
            self._end_of_turn_reset()

        # Game over check.
        if not self._game.tiles.highest():
            self._game.terminated = True
            self._game.explanation = "No tiles left on the table, game over."

    def step(
        self,
        action: tuple[int, int],
    ) -> tuple[dict[str, Any], int, bool, bool, dict[str, object]]:
        """Execute one turn of the game.

        The human player (agent) takes their turn by collecting a die and deciding whether to roll again or
        stop (taking/returning tiles). Then all bot players automatically execute their turns.

        Args:
            action: dice_index, roll_choice where:
                dice_index (0-5): Which die faces to collect.
                roll_choice: 0 to stop and claim a tile, 1 to roll again.

        Returns:
            - observation:
                'dice_collected', 'dice_rolled', 'tiles_table', 'tile_players'
            - reward: Score delta from a turn (worm tile value: -4 to +4)
            - terminated: Game over (no tiles remain on the table)
            - truncated: Illegal action, try again
            - info: Debug information with game state details.

        """
        # 1. Validate action.
        # 2. Process agent's turn (collect dice, take/return tile).
        # 3. Process all bots' turns.
        # 4. Check for game over.

        self._action = action
        reward = 0

        # Validate action and check if allowed before doing a step.
        self._game.terminated, self._game.truncated, self._game.explanation = self._game.action_checker.check(
            action,
        )

        # Illegal move
        if self._game.terminated or self._game.truncated:
            return (
                self._current_obs(),
                0,
                self._game.terminated,
                self._game.truncated,
                self._get_info(),
            )

        # Collect and roll the dice
        self._step_dice()

        # Action is to stop or failed attempt, get reward from step tiles.
        if self._action[ACTION_INDEX_ROLL] == ACTION_STOP or self._game.failed_attempt:
            reward = self._step_tiles()
            self._game.current_player_index = 1
            self._play_bot()  # Play the bots.
            self._game.current_player_index = 0

        # Game Over if no Tile is on the table anymore.
        if not self._game.tiles.highest():
            self._game.terminated = True
            self._game.explanation = "No tiles left on the table, game over."

        return (
            self._current_obs(),
            reward,
            self._game.terminated,
            self._game.truncated,
            self._get_info(),
        )

    def close(self) -> None:
        """Close the environment and release pygame resources.

        Must be called when finished with the environment to clean up pygame display and clock resources.

        """
        self._renderer.close()
