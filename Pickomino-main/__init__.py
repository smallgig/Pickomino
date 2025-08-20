# pickomino_env/__init__.py
from gymnasium.envs.registration import register

# Damit Nutzer die Klasse auch direkt importieren können:
from .pickomino_gym_env import PickominoEnv

# Beim Import des Pakets einmalig im Registry eintragen
def _auto_register():
    try:
        register(
            id="Pickomino-v0",
            entry_point="pickomino_env.pickomino_gym_env:PickominoEnv",
            kwargs={"num_players": 2},        # Default-Argument(e), kann man bei gym.make überschreiben
            max_episode_steps=200,            # optionales Episodenlimit
        )
    except Exception:
        # Falls bereits registriert (Doppelklick/Hot-Reload etc.) → still bleiben
        pass

_auto_register()

__all__ = ["PickominoEnv"]
