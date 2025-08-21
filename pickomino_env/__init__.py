# pickomino_env/__init__.py
from gymnasium.envs.registration import register

# Nur registrieren – Gymnasium lädt die Klasse erst bei gym.make()
try:
    register(
        id="Pickomino-v0",
        entry_point="pickomino_env.pickomino_gym_env:PickominoEnv",
        # kwargs={"num_players": 2},
        max_episode_steps=200,
    )
except Exception:
    # Schon registriert? Kein Problem.
    pass

# Optional: Lazy-Export, damit "from pickomino_env import PickominoEnv" funktioniert
def __getattr__(name):
    if name == "PickominoEnv":
        from .pickomino_gym_env import PickominoEnv as _P
        return _P
    raise AttributeError(name)

__all__ = ["PickominoEnv"]
