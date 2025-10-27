"""Gymnasium Environment for the game Pickomino, Heckmeck in German."""

from gymnasium.envs.registration import register

# Only register – Gymnasium loads the class with gym.make()
register(
    id="Pickomino-v0",
    entry_point="pickomino_env.pickomino_gym_env:PickominoEnv",
    # kwargs={"num_players": 2},
    max_episode_steps=200,
)
