# Pickomino

Implements the game [Pickomino](https://www.maartenpoirot.com/pickomino/play_pickomino_en) as an Environment with
a standard API for Reinforcement Learning.

# Pickomino Gymnasium Environment 🐛🎲

An environment conforming to the **Gymnasium** API for the dice game **Pickomino (Heckmeck am Bratwurmeck)**
Goal: train a Reinforcement Learning agent for optimal play (which dice to collect, when to stop).

## Content

* `pickomino_env/pickomino_gym_env.py` – your `PickominoEnv` class
* `pickomino_env/__init__.py` – **automatic registration** of the environment as `Pickomino-v0`
* `pyproject.toml` – Package-Metadata & dependencies

---

## Installation (developer mode)

```bash
# 1) Optional: virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 2) Dependencies & Package installation.
pip install -e .
```

> `-e` (editable) link to your work space - change in the code take effect immediately.

---

## Project Structure

```
pickomino-env/
├─ pyproject.toml
├─ README.md
└─ pickomino_env/
   ├─ __init__.py                 # registers "Pickomino-v0" when importet
   └─ pickomino_gym_env.py        # class PickominoEnv(gym.Env)
```

Optional: `rl_pickomino_qlearning.py` in the root (Training script).

---

## Environment usage

Registration happens automatically when importing from `pickomino_env`.

```python
import gymnasium as gym
import pickomino_env  # Important: this causes the registration.

env = gym.make("Pickomino-v0")  # kwargs overwrites the defaults.
obs, info = env.reset(seed=42)
print("Init ok. Example observation:", obs)
```

### Observations & Actions (current API)

* **Observation**: `obs = (dice_collected, dice_rolled)`
  Both are vectors of length 6 (index 0 = die face 1, index 1 = die face 2, ..., index 5 = worm).
* **Action**: `(face, roll_again)`

    * `face ∈ {0..5}` (5=worm) → collect all rolled dice with this face
    * `roll_again ∈ {0,1}` → 0 = **roll*, 1 = **stop**

---

## Rules (summary)

* 8 dice: `1..5` & **worm**. Worm count **5** towards the sum.
* You **have to** collect at least  **one worm** and **sum ≥ 21**, in order to pick a tile.
* When you stop rolling, you pick **the highest available tile ≤ sum**.
* (or steal the top tile from another player's stack if you have the exact sum).
* **A failed attempt** (no die can be collected or rules not followed): your top tile is returned to the table.
* If it is not the highest still available on the table, then the highest tile is turned face down.

---

## Typical issues and resolutions

1. **`ValueError: list.remove(x): x not in list`**
   Cause: `step_tiles()` tries to `tile_table.remove(sum)`.
   **Fix:** take **max(\[t for t in tile\_table if t ≤ sum])** only if you **stop** rolling
   (or **no die** left) *and* only with **at least one worm**.

2. **`legal_move` setting `self.terminated` to true, but gives local flag back.**
   Consistency: set and return **only local** variable **or** explicit `return self.terminated, self.truncated`.

3. **Observation-Space does not fit**
   For a clean gymnasium:

   ```python
   import numpy as np
   from gymnasium import spaces
   self.observation_space = spaces.Dict({
       "dice_collected": spaces.Box(low=0, high=8, shape=(6,), dtype=np.int64),
       "dice_rolled":    spaces.Box(low=0, high=8, shape=(6,), dtype=np.int64),
       "player":         spaces.Discrete(num_players),
   })
   ```

4. **Stop action**
   Model as discrete action in the Reinforcement Learning Agent (e.g. ID 12), map to `(face=0, roll_again=0)`.

---

## Tests (sanity check)

```python
import gymnasium as gym, pickomino_env
env = gym.make("Pickomino-v0")
obs, info = env.reset(seed=0)
for _ in range(5):
    action = (1, 1)   # Collect ones, keep rolling.
    obs, r, term, trunc, info = env.step(action)
    print("r=", r, "term=", term, "trunc=", trunc)
```

---

## Development

* Format: `ruff` / `black` recommended
* Lint: `pip install ruff black`
* Run: `ruff check . && black .`

---

## License

Select a License (e.g. MIT) and add a file `LICENSE`:

```
MIT License (c) 2025 Jarl, Robin
```

---

## Thanks

* Idea: **Pickomino (Heckmeck am Bratwurmeck)**
* Reinforcement Learning example: table Q-Learning (simple Baseline; for larger state spaces DQN is recommended.)
* Karsten and Tanja for their support.

---

**We wish success when training!**
