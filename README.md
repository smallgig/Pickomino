# Pickomino
Implements the game [Pickomino](https://www.maartenpoirot.com/pickomino/play_pickomino_en) as an Environment with a standard API for Reinforcement Learning.

# Pickomino Gymnasium Environment 🐛🎲

Ein **Gymnasium**-kompatibles Environment für das Würfelspiel **Pickomino (Heckmeck am Bratwurmeck)**
Ziel: Einen Agenten trainieren, der in diesem MDP optimale Entscheidungen trifft (Sorte wählen / stoppen).

## Inhalte

* `pickomino_env/pickomino_gym_env.py` – deine `PickominoEnv` Klasse
* `pickomino_env/__init__.py` – **automatische Registrierung** des Environments als `Pickomino-v0`
* `pyproject.toml` – Paket-Metadaten & Abhängigkeiten
---

## Installation (Entwicklungsmodus)

```bash
# 1) Optional: virtuelle Umgebung
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 2) Abhängigkeiten & Paket install.
pip install -e .
```

> `-e` (editable) verlinkt dein Arbeitsverzeichnis – Änderungen am Code wirken sofort.

---

## Projektstruktur

```
pickomino-env/
├─ pyproject.toml
├─ README.md
└─ pickomino_env/
   ├─ __init__.py                 # registriert "Pickomino-v0" beim Import
   └─ pickomino_gym_env.py        # class PickominoEnv(gym.Env)
```

Optional: `rl_pickomino_qlearning.py` im Root (Trainingsskript).

---

## Environment verwenden

Die Registrierung passiert automatisch beim Import von `pickomino_env`.

```python
import gymnasium as gym
import pickomino_env  # ⚠️ wichtig: löst die Registrierung aus

env = gym.make("Pickomino-v0")  # kwargs überschreiben Defaults
obs, info = env.reset(seed=42)
print("Init ok. Beispiel-Observation:", obs)
```

### Beobachtungen & Aktionen (derzeitige API)

* **Observation**: `obs = (dice_collected, dice_rolled)`
  Beide sind Längen-6-Vektoren (Index 0 = Wurm, 1..5 = Augen).
* **Action**: `(face, roll_again)`

  * `face ∈ {0..5}` (0=Wurm) → nimm alle geworfenen Würfel dieser Sorte
  * `roll_again ∈ {0,1}` → 0 = **stoppen**, 1 = **weiter würfeln**

> Hinweis: In der gelieferten Env sind die `observation_space`-Deklarationen noch `Discrete(6)`. Für algorithmische Stabilität empfiehlt sich **`Box(shape=(6,), dtype=int)`** o. **`MultiDiscrete([9]*6)`**. Das Beispiel-Training codiert die Observation intern selbst, daher läuft es auch so.

---

## Regeln (Kurzfassung)

* 8 Würfel: `1..5` & **Wurm** (W). Wurm zählt **5** zur Summe.
* Du **musst** mind. **einen Wurm** sammeln und **Summe ≥ 21**, um ein Plättchen zu nehmen.
* Beim Stoppen nimmst du das **höchste offene Plättchen ≤ Summe** (oder stiehlst exakt passendes Top-Plättchen eines Gegners).
* **Misswurf** (kein neues Gesicht wählbar oder Stop ohne Voraussetzungen): oberstes eigenes Plättchen zurück, höchstes offenes wird zusätzlich umgedreht.

---

## Typische Stolpersteine & Fixes

1. **`ValueError: list.remove(x): x not in list`**
   Ursache: `step_tiles()` versucht `tile_table.remove(sum)`.
   **Fix:** Nimm **max(\[t for t in tile\_table if t ≤ sum])** nur bei **Stop** (oder wenn **keine Würfel** übrig) *und* nur mit **mind. einem Wurm**.

2. **`legal_move` setzt `self.terminated`, gibt aber lokale Flags zurück**
   Konsistent machen: **nur lokale** Variablen setzen und zurückgeben **oder** explizit `return self.terminated, self.truncated`.

3. **Observation-Space passt nicht**
   Für Clean-Gym:

   ```python
   from gymnasium import spaces
   self.observation_space = spaces.Dict({
       "dice_collected": spaces.Box(low=0, high=8, shape=(6,), dtype=np.int64),
       "dice_rolled":    spaces.Box(low=0, high=8, shape=(6,), dtype=np.int64),
       "player":         spaces.Discrete(num_players),
   })
   ```

4. **Stop-Aktion**
   Im Agenten als eigene diskrete Aktion modelliert (z. B. ID 12), die auf `(face=0, roll_again=0)` gemappt wird.

---

## Tests (Schnellcheck)

```python
import gymnasium as gym, pickomino_env
env = gym.make("Pickomino-v0")
obs, info = env.reset(seed=0)
for _ in range(5):
    action = (1, 1)   # nimm „1er“, dann weiterwürfeln
    obs, r, term, trunc, info = env.step(action)
    print("r=", r, "term=", term, "trunc=", trunc)
```

---

## Entwicklung

* Format: `ruff` / `black` empfohlen
* Lint: `pip install ruff black`
* Run: `ruff check . && black .`

---

## Lizenz

Wähle eine Lizenz (z. B. MIT) und lege eine Datei `LICENSE` ab:

```
MIT License (c) 2025 Jarl,Robin, Tanja
```

---

## Danksagung

* Spielidee: **Heckmeck am Bratwurmeck**
* RL-Beispiel: tabellarisches Q-Learning (einfache Baseline; für größere Zustandsräume DQN empfehlen)

---

**Viel Erfolg beim Trainieren!**
