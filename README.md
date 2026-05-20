<div align="center">

# Pickomino-Env

**A Gymnasium Reinforcement Learning environment for the dice game Pickomino**<br/>
*Heckmeck am Bratwurmeck — by Reiner Knizia*

<br/>

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

<br/>

<img src="https://raw.githubusercontent.com/smallgig/Pickomino/main/assets/pickomino-demo.gif" width="560" alt="Animated demo of the Pickomino game played manually.">

<br/>
<br/>

[Quick Start](#installation) &nbsp;·&nbsp; [Play Manually](#play-manually) &nbsp;·&nbsp; [API Reference](#action-space) &nbsp;·&nbsp; [Contributing](#contributing)

<br/>

</div>

---

## Description

**Pickomino-Env** is a [Gymnasium](https://gymnasium.farama.org/)-compatible environment for training Reinforcement Learning agents to play **Pickomino** — a push-your-luck dice game designed by Reiner Knizia.

The game is played with 8 dice and 16 tiles numbered 21 to 36. Each tile carries one to four worm symbols, with higher-numbered tiles holding more worms. On each turn, a player rolls all available dice and must lock in one die face — then decide whether to keep rolling or stop and claim a tile. Each face can only be locked in once per turn. Roll a result where every face is already collected and the turn fails, costing the player their top tile.

What makes Pickomino strategically interesting is that the correct choice is rarely the obvious one. Locking in the highest-value dice is not always right. Deciding when to stop, which face to sacrifice, and whether to chase a high tile or settle for a safe one involves real probability reasoning — making it a strong candidate for Reinforcement Learning research.

> "The first dice game I've played which I think can seriously give Can't Stop a run for the money."
> — Larry Levy, [Playing the Odds — One Worm at a Time](https://boardgamegeek.com/thread/129610/pickomino-playing-the-odds-one-worm-at-a-time)

---

## Features

| | |
|---|---|
| **Gymnasium API** | Standard `reset` / `step` / `render` / `close` interface |
| **Push-your-luck mechanics** | Lock in die faces one at a time, decide when to stop before busting |
| **Non-trivial decisions** | Optimal play requires probability reasoning, not just greedy face selection |
| **Multi-player bots** | Play against 1–6 heuristic bot opponents |
| **Reproducible episodes** | Full seed support via `env.reset(seed=42)` |
| **Three render modes** | `None` (headless), `"human"` (pygame window), `"rgb_array"` (recording) |
| **SB3 compatible** | Dict observation space works with Stable-Baselines3 and other RL libraries |
| **Strict type safety** | Fully annotated, verified with Pyright and mypy in strict mode |
| **95%+ test coverage** | Enforced in CI across Python 3.10–3.14 |

---

## Installation

> Requires Python 3.10–3.14. A virtual environment is recommended.

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows cmd.exe
.venv\Scripts\activate.bat

# Windows Git Bash
source .venv/Scripts/activate

pip install pickomino-env
```

Verify the installation:

```bash
pickomino-play
```

---

## Quick Start

```python
import gymnasium as gym

env = gym.make("Pickomino-v0", number_of_bots=2)
obs, info = env.reset(seed=42)

terminated = False
truncated = False
total_reward = 0

while not terminated and not truncated:
    action = env.action_space.sample()          # (die_face, roll_or_stop)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if truncated:
        print(f"Invalid action: {info['explanation']}")

print(f"Episode finished. Total reward: {total_reward}")
env.close()
```

---

## Play Manually

Playing a few games by hand is the fastest way to understand the rules and the strategic depth before training an agent. Launch the pygame GUI:

```bash
# One bot (default)
pickomino-play

# Up to six bots
pickomino-play --number-of-bots=3
```

To adjust bot speed, change `RENDER_DELAY` in `constants.py`. A higher value slows bots down; lower speeds them up.

```python
RENDER_DELAY: Final[float] = 2
```

---

## Action Space

The action space is `MultiDiscrete([6, 2])`. `step()` accepts both a NumPy ndarray and a plain Python tuple.

```
action = (die_face, action_type)
```

| Dimension | Value | Meaning |
|---|---|---|
| `die_face` | `0` – `4` | Lock in all dice showing 1 through 5 eyes |
| | `5` | Lock in all worm dice |
| `action_type` | `0` | Roll the remaining dice again |
| | `1` | Stop and claim a tile |

---

## Observation Space

The observation is a `dict` with four keys, returned at every `reset()` and `step()`:

| Key | Shape | Range | Description |
|---|---|---|---|
| `dice_collected` | `(6,)` | `[0, 8]` | Count of each die face locked in this turn |
| `dice_rolled` | `(6,)` | `[0, 8]` | Count of each die face in the current roll |
| `tiles_table` | `(16,)` | `{0, 1}` | Binary — which tiles (21–36) are still on the table |
| `tile_players` | `(n_players,)` | `[0, 36]` | Top tile held by each player (`0` = none) |

There are 8 dice, each with faces 1–5 plus a worm. The worm scores 5 points — the same as the 5-eye face, not 6. The 16 tiles are numbered 21–36 and carry 1–4 worms each in groups of four. At least one worm must be locked in to claim any tile.

---

## Reward Function

The agent receives a reward at the end of each turn when a tile is claimed or returned.

| Outcome | Reward | Description |
|---|---|---|
| Claim tile 33–36 | `+4` | Four-worm tiles claimed successfully |
| Claim tile 29–32 | `+3` | Three-worm tiles claimed successfully |
| Claim tile 25–28 | `+2` | Two-worm tiles claimed successfully |
| Claim tile 21–24 | `+1` | One-worm tiles claimed successfully |
| Steal an opponent's tile | `+1` to `+4` | Worm value of the stolen tile |
| Failed attempt — tile returned | `-1` to `-4` | Worm value of the returned tile, applied as a penalty |
| Failed attempt — empty stack | `0` | No tile to return; no reward change |
| Opponent steals your tile | `0` | No penalty when a bot steals from the agent |

> The total reward at the end of a game can exceed the final worm score, because stolen tiles add to cumulative reward without reducing the agent's stack.

---

## Episode Termination

### Terminated

The episode ends naturally when **no tiles remain on the table**.

```
terminated = True   # Game over — no tiles left
```

### Truncated

Truncation occurs when the agent submits an **illegal action**. The episode is not over — submit a valid action on the next `step()` call.

```
truncated = True    # Illegal action — try again
```

Common causes:

- Selecting a die face not present in the current roll
- Selecting a die face already locked in this turn
- Choosing to roll again when no dice remain

> Out-of-range actions (outside `[0–5]` or `[0–1]`) raise a `ValueError` without changing episode state.

### Failed Attempt

A bust occurs when the agent cannot lock in any new die face — for example, when every rolled face has already been collected. The agent's **top tile is returned** to the table and a negative reward is applied. If the stack is empty, the reward is `0`. **The episode continues.**

---

## Starting State

```
dice_collected = [0, 0, 0, 0, 0, 0]
dice_rolled    = [3, 0, 1, 2, 0, 2]   # random roll, example sum = 8
tiles_table    = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
tile_players   = [0, 0, 0]             # example with number_of_bots = 2
```

---

## Info Dictionary

Returned at every `step()`. Intended for **debugging and logging only** — not for learning.

| Key | Type | Description |
|---|---|---|
| `dice_collected` | `list[int]` | Die face counts locked in this turn |
| `dice_rolled` | `list[int]` | Die face counts in the current roll |
| `terminated` | `bool` | Whether the episode has ended |
| `truncated` | `bool` | Whether the last action was illegal |
| `tiles_table_vec` | `ndarray[int8]`, shape `(16,)` | Binary tile availability vector |
| `smallest_tile` | `int` | Lowest-numbered tile still on the table |
| `explanation` | `str` | Reason for termination, truncation, or bust |
| `player_stack` | `list[int]` | All tiles currently held by the agent |
| `player_score` | `int` | Agent's current worm score |
| `current_player_index` | `int` | Index of the active player |
| `bot_scores` | `list[int]` | Scores of all bots, in order |

---

## Arguments

| Parameter | Type | Default | Description |
|---|---|---|---|
| `number_of_bots` | `int` | `1` | Number of bot opponents (1–6) |
| `render_mode` | `str \| None` | `None` | `None` · `"human"` · `"rgb_array"` |

---

## Bot Heuristic

The bots follow a fixed strategy inspired by [Frozen Fractal's analysis](https://frozenfractal.com/blog/2015/5/3/how-to-win-at-pickomino/):

1. **Highest contribution first** — lock in the face where `count × value` is greatest. Worms count as 5.
2. **Tie-breaking** — prefer worms over 5s. If still tied, prefer the face with fewer dice to preserve future rolls.
3. **Worm priority on roll 3 and beyond** — if no dice have been locked in yet and this is the third roll or later, always take worms if available.
4. **Stop as soon as a tile is reachable** — once the running total meets or exceeds the smallest available tile value and a worm is locked in, stop.

---

## Differences from the Physical Game

| Aspect | Physical Game | This Environment |
|---|---|---|
| Failed attempt | Highest tile turned face-down | Highest tile removed from the table |
| Tile selection | Player chooses which tile to take | Best reachable tile taken automatically |
| Stealing | Optional | Always performed when possible |
| Win condition | Most worms (tie broken by highest tile) | Use total reward as your metric when training |
| Stack height | Visible to all players | Not included in the observation |

---

## Security & Bug Bounty

Found a bug? Valid reports are rewarded with a **physical copy of the Pickomino board game**.
See [SECURITY.md](https://github.com/smallgig/Pickomino/blob/main/SECURITY.md) for scope, timelines, and reporting instructions.

---

## Contributing

Contributions are welcome. The project runs two-week sprints with issues assigned to contributors.

1. Browse or open an issue on [GitHub Issues](https://github.com/smallgig/Pickomino/issues)
2. Create a branch using the format `<issue-number>-<brief-description>`
3. Run `pre-commit run --all-files` before pushing
4. Open a Pull Request from your branch to `main`

See [CONTRIBUTING.md](https://github.com/smallgig/Pickomino/blob/main/CONTRIBUTING.md) for the full workflow, code style requirements, and definition of done.

---

## Resources

- **Game Rules** — [Pickomino Rulebook](https://github.com/smallgig/Pickomino/blob/main/pickomino-rulebook.pdf)
- **Play Online** — [Maarteen Poirot's Pickomino](https://www.maartenpoirot.com/pickomino/)
- **Play on Board Game Arena** — [Pickomino with Elo](https://boardgamearena.com/14/pickomino?table=818236942)
- **Strategy Discussion** — [Playing the Odds — One Worm at a Time](https://boardgamegeek.com/thread/129610/pickomino-playing-the-odds-one-worm-at-a-time)
- **Bot Strategy** — [How to Win at Pickomino](https://frozenfractal.com/blog/2015/5/3/how-to-win-at-pickomino/)
- **Gymnasium Docs** — [gymnasium.farama.org](https://gymnasium.farama.org/)

---

## Contact

Maintained by [smallgig](https://github.com/smallgig).
For questions or ideas, open an issue with the label `question`.

---

## License

MIT License — see [LICENSE](LICENSE) for details.