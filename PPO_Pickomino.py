import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.spaces.utils import flatten, flatten_space
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
import os

# -------- Debug-Validator (prüft ROH-Obs vor dem Flatten) --------
class ValidateObs(gym.Wrapper):
    def _check(self, obs, space, path=""):
        if isinstance(space, spaces.Dict):
            assert isinstance(obs, dict), f"{path}: erwartet dict, bekam {type(obs)}"
            for k, sp in space.spaces.items():
                assert k in obs, f"{path}: key fehlt: {k}"
                self._check(obs[k], sp, f"{path}.{k}" if path else k)
        elif isinstance(space, spaces.Tuple):
            assert isinstance(obs, (tuple, list)), f"{path}: erwartet tuple/list, bekam {type(obs)}"
            assert len(obs) == len(space.spaces), f"{path}: Länge passt nicht"
            for i, sp in enumerate(space.spaces):
                self._check(obs[i], sp, f"{path}[{i}]")
        elif isinstance(space, (spaces.Box, spaces.MultiBinary, spaces.MultiDiscrete)):
            assert not isinstance(obs, spaces.Space), f"{path}: VALUE IST SPACE {type(obs)}"
        elif isinstance(space, spaces.Discrete):
            assert not isinstance(obs, spaces.Space), f"{path}: VALUE IST SPACE {type(obs)}"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._check(obs, self.env.observation_space)
        return obs, info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        self._check(obs, self.env.observation_space)
        return obs, r, terminated, truncated, info

# -------- Actions: Tuple(Discrete,...) -> MultiDiscrete --------
class TupleToMultiDiscrete(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        if isinstance(env.action_space, spaces.Tuple) and all(isinstance(s, spaces.Discrete) for s in env.action_space.spaces):
            self.dims = [s.n for s in env.action_space.spaces]
            self.action_space = spaces.MultiDiscrete(self.dims)
        else:
            self.dims = None
            self.action_space = env.action_space
    def action(self, a):
        return tuple(int(x) for x in a) if self.dims else a

# -------- Null-Werte bauen (rekursiv) --------
def zeros_from_space(sp: spaces.Space):
    if isinstance(sp, spaces.Box):
        return np.zeros(sp.shape, dtype=sp.dtype)
    if isinstance(sp, spaces.Discrete):
        return 0
    if isinstance(sp, spaces.MultiBinary):
        return np.zeros(sp.n, dtype=np.int8)
    if isinstance(sp, spaces.MultiDiscrete):
        return np.zeros(len(sp.nvec), dtype=np.int64)
    if isinstance(sp, spaces.Dict):
        return {k: zeros_from_space(sub) for k, sub in sp.spaces.items()}
    if isinstance(sp, spaces.Tuple):
        return tuple(zeros_from_space(sub) for sub in sp.spaces)
    raise TypeError(f"zeros_from_space: nicht unterstützt: {type(sp)}")

def _sanitize_discrete(space: spaces.Discrete, obs):
    v = int(obs)
    lo = int(space.start)
    hi = lo + int(space.n) - 1
    if not (lo <= v <= hi):
        # auf gültigen Bereich abbilden (Modulo), dann wieder verschieben
        if space.n > 0:
            v = lo + ((v - lo) % int(space.n))
        else:
            v = lo
    return v

def _sanitize_multidiscrete(space: spaces.MultiDiscrete, obs):
    arr = np.asarray(obs, dtype=np.int64).ravel()
    n = np.asarray(space.nvec, dtype=np.int64).ravel()
    # Länge korrigieren, falls nötig
    if arr.size != n.size:
        arr = np.zeros_like(n)
    # in gültigen Bereich clippen: [0, n_i-1]
    n_safe = np.maximum(n, 1)
    arr = np.mod(arr, n_safe)
    return arr.astype(np.int64)

def _sanitize_multibinary(space: spaces.MultiBinary, obs):
    arr = np.asarray(obs, dtype=np.int8).ravel()
    if arr.size != space.n:
        arr = np.zeros(space.n, dtype=np.int8)
    # nur {0,1} erlauben
    arr = (arr != 0).astype(np.int8)
    return arr

def sanitize_obs(space: spaces.Space, obs):
    # Falls versehentlich ein Space-Objekt als Wert kommt -> Nullwerte
    if isinstance(obs, spaces.Space):
        return zeros_from_space(space)

    if isinstance(space, spaces.Dict):
        if not isinstance(obs, dict):
            return zeros_from_space(space)
        return {k: sanitize_obs(sp_k, obs.get(k, zeros_from_space(sp_k)))
                for k, sp_k in space.spaces.items()}

    if isinstance(space, spaces.Tuple):
        if not isinstance(obs, (tuple, list)) or len(obs) != len(space.spaces):
            return zeros_from_space(space)
        return tuple(sanitize_obs(sp_k, v) for sp_k, v in zip(space.spaces, obs))

    if isinstance(space, spaces.Discrete):
        return _sanitize_discrete(space, obs)

    if isinstance(space, spaces.MultiDiscrete):
        return _sanitize_multidiscrete(space, obs)

    if isinstance(space, spaces.MultiBinary):
        return _sanitize_multibinary(space, obs)

    if isinstance(space, spaces.Box):
        # in dtype gießen; bei Shape-Mismatch fallback auf zeros in gleicher Shape
        arr = np.asarray(obs, dtype=space.dtype)
        try:
            arr = arr.reshape(space.shape)
        except Exception:
            arr = np.zeros(space.shape, dtype=space.dtype)
        # optional: in Bounds clippen, wenn finite
        if np.all(np.isfinite(space.low)) and np.all(np.isfinite(space.high)):
            arr = np.clip(arr, space.low, space.high)
        return arr

    return obs


# -------- Sicheres Flatten zu 1D-Box --------
class SafeFlattenToBox(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._S = env.observation_space
        self.observation_space = flatten_space(self._S)
    def observation(self, obs):
        obs = sanitize_obs(self._S, obs)
        return flatten(self._S, obs)

# -------- TensorBoard-Callback (zusätzliche Scalars) --------
class TBCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._updates = 0
    def _on_step(self) -> bool:
        self._updates += 1
        # Lernrate loggen (falls Schedules genutzt werden)
        lr = float(self.model.lr_schedule(self.model._current_progress_remaining))
        self.logger.record("opt/lr", lr)
        self.logger.record("train/updates", self._updates)
        return True


# ---- Env-Fabrik: Action -> SafeFlatten -> Monitor (ohne ValidateObs) ----
def make_env(seed: int = 0):
    e = gym.make("Pickomino-v0")
    e = TupleToMultiDiscrete(e)  # Tuple(Discrete,...) -> MultiDiscrete
    e = SafeFlattenToBox(e)      # rekursiv sanitisieren + 1D-Box flatten
    e = Monitor(e)
    e.reset(seed=seed)
    return e

env = make_env(0)


check_env(env, warn=True)


model = PPO("MlpPolicy", env, tensorboard_log="runs", verbose=1)
model.learn(10_000, tb_log_name="PickominoPPO")
from stable_baselines3.common.callbacks import EvalCallback

eval_env = make_env(seed=123)
eval_cb = EvalCallback(
    eval_env, eval_freq=5000, n_eval_episodes=20,
    deterministic=True, best_model_save_path="models", log_path="eval"
)
model.learn(200_000, tb_log_name="PickominoPPO", callback=[TBCallback(), eval_cb])
