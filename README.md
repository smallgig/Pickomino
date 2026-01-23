# Pickomino-Env

[![PyPI version](https://img.shields.io/pypi/v/pickomino-env.svg)](https://pypi.org/project/pickomino-env/)
[![Python 3.8-3.13](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
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

## Description

An environment conforming to the **Gymnasium** API for the dice game **Pickomino (Heckmeck am Bratwurmeck)**
Goal: train a Reinforcement Learning agent for optimal play. Meaning, decide which face of the dice to collect,
when to roll and when to stop.

## Action Space

The Action space is a tuple with two integers.
Tuple (int, int)

Action = [dice_face (0-5), action_type (0=roll, 1=stop)].

- 0-5: Face of the dice, which you want to take, where:
    - 0 -> face 1
    - 1 -> face 2
    - 2 -> face 3
    - 3 -> face 4
    - 4 -> face 5
    - 5 -> face worm


- 0-1: Roll (0) or stop (1).

## Observation Space

The observation is a `dict` with shape `(4,)` with the values corresponding to the following: dice, table and player.

| Observation    | Min | Max | Shape             |
|----------------|-----|-----|-------------------|
| dice_collected | 0   | 8   | (6,)              |
| dice_rolled    | 0   | 8   | (6,)              |
| tiles_table    | 0   | 1   | (16,)             |
| tile_players   | 0   | 36  | number_of_players |

**Note:** There are eight dice to roll and collect. A die has six sides with the number of eyes one through
five, but a worm instead of a six.
The values correspond to the number of eyes, with the worm also having the value five (and not six!).
The 16 tiles are numbered 21 to 36 and have worm values from one to four in spread in four groups.
The game is for two to seven players. Here your Reinforcement Learning Agent is the first player. The
other players are computer bots.
The bots play, according to a heuristic. When you create the environment,
you have to define the number of bots.

For a more detailed description of the rules, see the file pickomino-rulebook.pdf.
You can play the game online here: https://www.maartenpoirot.com/pickomino/.
The heuristic used by the bots is described here: https://frozenfractal.com/blog/2015/5/3/how-to-win-at-pickomino/.

## Rewards

The goal is to collect tiles in a stack. The winner is the player, which at the end of the game has the most worms
on her tiles. For the Reinforcement Learning Agent a reward equal to the value
(worms) of a tile is given when the tile is picked. For a failed attempt
(see rulebook), a corresponding negative reward is given. When a bot steals your
tile, no negative reward is given. Hence, the total reward at the end of the game
can be greater than the score.

## Starting State

* `dice_collected` = [0, 0, 0, 0, 0, 0].
* `dice_rolled` = [3, 0, 1, 2, 0, 2] Random dice, sum = 8.
* `tiles_table` = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1].
* `tile_players` = [0, 0, 0] (with number_of_bots = 2).

## Episode End

The episode ends if one of the following occurs:

1. Termination: If there are no more tiles to take on the table = Game Over.
2. Termination: Action out of allowed range [0–5, 0-1].

### Truncation

Truncation: Attempt to break the rules, the game continues, and you have to give a new valid action.

### Failed Attempt

Note that a Failed Attempt means: If a tile is present, put it back on the table and get a negative reward.
However, the game continues, so the Episode does not end.

## Arguments

These must be specified.

| Parameter        | Type        | Default | Description                                                                                |
|------------------|-------------|---------|--------------------------------------------------------------------------------------------|
| `number_of_bots` | int         | --      | Number of bot opponents (1-6) you want to play against                                     |
| `render_mode`    | str or None | None    | Visualization mode:<br/>None (training),<br/>"human" (display), or "rgb_array" (recording) |

## Setup

`pip install pickomino-env`

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
