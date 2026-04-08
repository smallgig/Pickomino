# Pickomino-Env

[![PyPI version](https://img.shields.io/pypi/v/pickomino-env.svg)](https://pypi.org/project/pickomino-env/)
[![CI](https://github.com/smallgig/Pickomino/actions/workflows/python-package.yml/badge.svg)](https://github.com/smallgig/Pickomino/actions/workflows/python-package.yml)
[![Publish](https://github.com/smallgig/Pickomino/actions/workflows/python-publish.yml/badge.svg)](https://github.com/smallgig/Pickomino/actions/workflows/python-publish.yml)
[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
[![Type hints: Pyright](https://img.shields.io/badge/type%20hints-Pyright-brightgreen.svg)](https://github.com/microsoft/pyright)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gymnasium](https://img.shields.io/badge/API-Gymnasium-brightgreen)](https://gymnasium.farama.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pydocstyle](https://img.shields.io/badge/docstrings-pydocstyle-brightgreen)](http://www.pydocstyle.org/)
[![Code complexity: radon](https://img.shields.io/badge/code%20complexity-radon-brightgreen)](https://radon.readthedocs.io/)
[![Complexity: xenon](https://img.shields.io/badge/complexity-xenon-brightgreen)](https://xenon.readthedocs.io/)
[![Pylint](https://img.shields.io/badge/pylint-checked-brightgreen)](https://pylint.pycqa.org/)
[![Type hints: mypy](https://img.shields.io/badge/type%20hints-mypy-brightgreen.svg)](http://mypy-lang.org/)
[![pytest: 95%+ coverage](https://img.shields.io/badge/pytest-95%25%2B%20coverage-brightgreen)](https://pytest.org/)

<div align="center">
    <img src="https://raw.githubusercontent.com/smallgig/Pickomino/main/assets/pickomino-demo.gif" width="500" alt="Animated demo of the Pickomino game played manually.">
</div>

## Description

An environment conforming to the **Gymnasium** API for the dice game **Pickomino (Heckmeck am Bratwurmeck)**
Goal: train a Reinforcement Learning agent for optimal play. Meaning, decide which face of the dice to collect,
when to roll and when to stop.

## Action Space
The action space is a `Tuple` of two integers:

`action = [die_face (0–5), action_type (0=roll, 1=stop)]`

| Index | die_face | action_type |
|-------|-----------|-------------|
| 0–5   | Die face to collect: 0→1 eye, 1→2 eyes, 2→3 eyes, 3→4 eyes, 4→5 eyes, 5→worm | — |
| 0–1   | — | 0 = roll again, 1 = stop and take a tile |

## Observation Space
The observation is a `dict` with 4 keys:

| Key            | Min | Max | Shape             |
|----------------|-----|-----|-------------------|
| dice_collected | 0   | 8   | (6,)              |
| dice_rolled    | 0   | 8   | (6,)              |
| tiles_table    | 0   | 1   | (16,)             |
| tile_players   | 0   | 36  | number_of_players |

There are eight dice, each with faces 1–5 plus a worm. The worm scores 5 points
— the same as the 5-eye face. It is not a sixth distinct value.

The 16 tiles are numbered 21–36 with worm values from one to four, spread across
four groups. The game supports two to seven players. The RL agent is player one;
the remaining players are computer bots controlled by a heuristic. The number of
bots is set when creating the environment.

The goal is to collect tiles in a stack. The winner is the player, which at the end of the game has the most worms
on her tiles. For the Reinforcement Learning Agent a reward equal to the value
(worms) of a tile is given when the tile is picked. For a failed attempt
(see rulebook), a corresponding negative reward is given. When a bot steals your
tile, no negative reward is given. Hence, the total reward at the end of the game
can be greater than the score.

For the full rules see the [Pickomino rulebook](https://github.com/smallgig/Pickomino/raw/main/pickomino-rulebook.pdf) or
[play online](https://www.maartenpoirot.com/pickomino/).
To try the environment manually, see [Play manually](#play-manually).
The bot heuristic is described [here](https://frozenfractal.com/blog/2015/5/3/how-to-win-at-pickomino/).

## Info Dictionary

The `info` dictionary is returned at every step. It is intended for debugging and
logging, not for learning.

| Key                   | Type        | Description                                              |
|-----------------------|-------------|----------------------------------------------------------|
| `dice_collected`      | `list[int]` | Counts of each die face collected this turn              |
| `dice_rolled`         | `list[int]` | Counts of each die face in the current roll              |
| `terminated`          | `bool`      | Whether the episode has terminated                       |
| `truncated`           | `bool`      | Whether the game was truncated due to the last action    |
| `tiles_table_vec`     | `list[int]` | Binary vector of tiles currently available on the table  |
| `smallest_tile`       | `int`       | Lowest-numbered tile still on the table                  |
| `explanation`         | `str`       | Description of the reason for a failed attempt           |
| `player_stack`        | `list[int]` | All tiles currently held by the agent                    |
| `player_score`        | `int`       | Agent's current score (sum of worm values)               |
| `current_player_index`| `int`       | Index of the player whose turn it is                     |
| `bot_scores`          | `list[int]` | Scores of all bots, in order                             |

## Starting State

* `dice_collected` = [0, 0, 0, 0, 0, 0].
* `dice_rolled` = [3, 0, 1, 2, 0, 2] Random dice, sum = 8.
* `tiles_table` = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1].
* `tile_players` = [0, 0, 0] (with number_of_bots = 2).

## Episode End

Termination occurs when there are no more tiles to take on the table — Game Over.

### Truncation
Truncation occurs when the agent attempts an illegal move (e.g. taking a tile
not currently available). The game continues and a new valid action is required.

### Invalid Actions
Out-of-range actions (outside [0–5] or [0–1]) raise a `ValueError` and do not
affect the episode state.

### Failed Attempt
A Failed Attempt occurs when the agent fails to secure a tile. If the agent has
a stack of already picked tiles of at least one, then the top tile is returned to the table and a negative reward is applied.
If the stack is empty, nothing happens and the reward is zero. The game continues
— the episode does not end.

## Arguments

These must be specified.

| Parameter        | Type        | Default | Description                                                                                |
|------------------|-------------|---------|--------------------------------------------------------------------------------------------|
| `number_of_bots` | int         | 1       | Number of bot opponents (1-6) you want to play against                                     |
| `render_mode`    | str or None | None    | Visualization mode:<br/>None (training),<br/>"human" (display), or "rgb_array" (recording) |

## Setup

`pip install pickomino-env`

## Play manually

Playing a few games manually is a great way to understand the rules and game dynamics
before training a Reinforcement Learning agent. Launch the game with the pygame GUI:

```
pickomino-play
```

To play against more bots:

```
pickomino-play --number-of-bots=3
```

Valid range: 1-6 bots.

## Usage example

```python
import gymnasium as gym

# Create environment
env = gym.make("Pickomino-v0", render_mode="human", number_of_bots=2)

# Reset and get initial observation
obs, info = env.reset(seed=42)

# Run one episode
terminated = False
truncated = False
total_reward = 0

while not terminated and not truncated:
    # Agent selects action: (dice_face, roll_choice)
    action = env.action_space.sample()  # Random action for demo

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if truncated:
        print(f"Invalid action: {info['explanation']}")
        break

print(f"Episode finished. Total reward: {total_reward}")
env.close()
```

## Resources

- **Game Rules:** [Pickomino Rulebook](https://github.com/smallgig/Pickomino/blob/main/pickomino-rulebook.pdf)
- **Play Online:** [Maarteen Poirot's Pickomino](https://www.maartenpoirot.com/pickomino/)
- **Bot Strategy:** [How to Win at Pickomino](https://frozenfractal.com/blog/2015/5/3/how-to-win-at-pickomino/)
- **Repository:** [smallgig/Pickomino](https://github.com/smallgig/Pickomino)
- **Gymnasium:** [https://gymnasium.farama.org/](https://gymnasium.farama.org/)

## License

MIT License. See [LICENSE](LICENSE) for details.
