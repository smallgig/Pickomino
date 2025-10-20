# qlearn_pickomino_tb.py
import os
import pickle
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter  # leichtgewichtig, kein PyTorch nötig

# ------------------------ Hilfsfunktionen ------------------------

TILES = list(range(21, 37))  # 21..36


def worms_for_tile(v: int) -> int:
    if 21 <= v <= 36:
        return (v - 21) // 4 + 1
    return 0


def score_of(dc: List[int]) -> int:
    # Wurm (Index 0) zählt 5, Augen i zählen i
    return int(dc[0] * 5 + sum(i * dc[i] for i in range(1, 6)))


def tiles_mask(tt: Dict[int, bool | int]) -> int:
    """Kompakte Bitmaske (16 Bits) aus tiles_table. True oder int>0 = liegt auf dem Tisch."""
    m = 0
    if not isinstance(tt, dict):
        for v in TILES:
            m |= 1 << (v - 21)
        return m
    for v in TILES:
        val = tt.get(v, False)
        is_on_table = (val is True) or (isinstance(val, int) and val > 0)
        if is_on_table:
            m |= 1 << (v - 21)
    return m


def obs_to_state_key(obs: dict) -> tuple:
    """Robuste, hashbare Zustandsrepräsentation für die Q-Table."""
    dc = tuple(int(x) for x in obs.get("dice_collected", [0, 0, 0, 0, 0, 0]))
    dr = tuple(int(x) for x in obs.get("dice_rolled", [0, 0, 0, 0, 0, 0]))
    mask = tiles_mask(obs.get("tiles_table", {}))

    # tiles_player kann fehlen/ein int/eine Liste sein
    tp = obs.get("tiles_player", 0)
    if isinstance(tp, list):
        top_tile = tp[-1] if tp else 0
    else:
        try:
            top_tile = int(tp)
        except Exception:
            top_tile = 0
    topW = worms_for_tile(top_tile)
    return (dc, dr, mask, topW)


def legal_actions(obs: Dict) -> List[Tuple[int, int]]:
    """
    Erzeuge legale Aktionen (face, roll_or_stop).
    - Gesicht legal, wenn im Wurf >0 und noch nicht gesammelt.
    - (f,0) = weiter würfeln ist immer erlaubt (nach Pick von f).
    - (f,1) = STOP nur, wenn nach dem Pick Wurm vorhanden und Summe >= 21.
    Wenn kein Gesicht legal (no-throw), fallback [(0,1)] – Env behandelt Bust.
    """
    dc = [int(x) for x in obs.get("dice_collected", [0] * 6)]
    dr = [int(x) for x in obs.get("dice_rolled", [0] * 6)]
    faces = [i for i in range(6) if dr[i] > 0 and dc[i] == 0]
    if not faces:
        return [(0, 1)]

    acts = []
    for f in faces:
        acts.append((f, 0))  # weiter würfeln nach Pick
        dc_new = dc.copy()
        dc_new[f] += dr[f]
        s_new = score_of(dc_new)
        stop_ok = dc_new[0] > 0 and s_new >= 21
        if stop_ok:
            acts.append((f, 1))
    return acts


def epsilon_greedy(
    Qs: Dict[Tuple, float], actions: List[Tuple[int, int]], eps: float
) -> Tuple[int, int]:
    if not actions:
        return (0, 1)
    if random.random() < eps:
        return random.choice(actions)
    best_a = max(actions, key=lambda a: Qs.get(a, 0.0))
    return best_a


def moving_average(values: List[float], window: int = 100) -> List[float]:
    if window <= 1:
        return values[:]
    out = []
    q = deque(maxlen=window)
    for v in values:
        q.append(v)
        out.append(sum(q) / len(q))
    return out


# ------------------------ Q-Learning ------------------------


def train_qlearning(
    env_id: str = "Pickomino-v0",
    episodes: int = 1000,
    alpha: float = 0.15,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.999,
    seed: int | None = None,
    save_path: str = "qtable_pickomino.pkl",
    log_dir: str = "runs/pickomino",
    plot_path: str = "learning_curve.png",
    ma_window: int = 200,
):
    env = gym.make(env_id)
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()

    # Q: dict[state_key][action] = value
    Q: Dict[Tuple, Dict[Tuple[int, int], float]] = defaultdict(dict)

    writer = SummaryWriter(log_dir=log_dir)
    eps = eps_start

    episode_returns: List[float] = []
    episode_steps: List[int] = []

    for ep in range(1, episodes + 1):
        if seed is not None:
            obs, info = env.reset(seed=seed + ep)
        else:
            obs, info = env.reset()

        done = False
        ep_return = 0.0
        steps = 0

        while not done:
            s = obs_to_state_key(obs)
            acts = legal_actions(obs)
            a = epsilon_greedy(Q[s], acts, eps)

            obs2, reward, terminated, truncated, info = env.step(list(a))

            s2 = obs_to_state_key(obs2)
            acts2 = legal_actions(obs2)

            # TD-Target
            if terminated or truncated or not acts2:
                target = reward
            else:
                max_next = max((Q[s2].get(a2, 0.0) for a2 in acts2), default=0.0)
                target = reward + gamma * max_next

            # Q-Update
            old = Q[s].get(a, 0.0)
            Q[s][a] = old + alpha * (target - old)

            obs = obs2
            ep_return += reward
            steps += 1
            done = bool(terminated or truncated)

        # Loggen
        writer.add_scalar("Return/episode", ep_return, ep)
        writer.add_scalar("Epsilon/episode", eps, ep)
        writer.add_scalar("Steps/episode", steps, ep)

        # (Optional) Verteilung der Q-Werte loggen
        if ep % 100 == 0:
            all_q = [v for d in Q.values() for v in d.values()]
            if all_q:
                writer.add_histogram("Q/values", all_q, ep)

        # Epsilon-Decay
        eps = max(eps_end, eps * eps_decay)

        # Stats fürs Plotten sammeln
        episode_returns.append(ep_return)
        episode_steps.append(steps)

        if ep % 200 == 0:
            ma = moving_average(episode_returns, ma_window)
            print(
                f"[Ep {ep:5d}] return={ep_return:+.2f}  avg({ma_window})≈{ma[-1]:+.2f}  eps={eps:.3f}  steps={steps}"
            )

    # Speichern
    with open(save_path, "wb") as f:
        pickle.dump(dict(Q), f)
    print(f"Q-Table gespeichert unter: {save_path}")

    # Lernkurven-Plot
    ma = moving_average(episode_returns, ma_window)
    plt.figure()
    plt.plot(
        range(1, len(episode_returns) + 1), episode_returns, label="Return/Episode"
    )
    plt.plot(range(1, len(ma) + 1), ma, label=f"Moving Avg (window={ma_window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Q-Learning Pickomino-v0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    print(f"Lernkurven-Plot gespeichert unter: {os.path.abspath(plot_path)}")

    writer.close()
    env.close()
    return Q, episode_returns


# ------------------------ Auswertung (greedy) ------------------------


def run_greedy_episode(
    env_id: str, Q: Dict, render: bool = False, seed: int | None = None
) -> float:
    env = gym.make(env_id)
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()

    done = False
    ret = 0.0
    steps = 0
    while not done:
        s = obs_to_state_key(obs)
        acts = legal_actions(obs)
        a = max(acts, key=lambda x: Q.get(s, {}).get(x, 0.0)) if acts else (0, 1)
        obs, reward, terminated, truncated, info = env.step(list(a))
        ret += reward
        steps += 1
        if render:
            print(
                f"[{steps:03d}] a={a}  r={reward:+.1f}  sum={info.get('self._sum')}  term={terminated} trunc={truncated}"
            )
        done = bool(terminated or truncated)
    env.close()
    return ret


# ------------------------ Main ------------------------

if __name__ == "__main__":
    print(
        "Training Q-Learning auf Pickomino-v0 (mit TensorBoard & Lernkurven-Plot) ..."
    )
    Q, rets = train_qlearning(
        env_id="Pickomino-v0",
        episodes=10000,  # je nach Geduld erhöhen
        alpha=0.15,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.999,
        seed=None,
        save_path="qtable_pickomino.pkl",
        log_dir="runs/pickomino",
        plot_path="learning_curve.png",
        ma_window=200,
    )

    print("\nTest (greedy policy, 1 Episode):")
    total = run_greedy_episode("Pickomino-v0", Q, render=True)
    print(f"Greedy-Return: {total:+.1f}")
