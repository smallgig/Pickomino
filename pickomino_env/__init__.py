"""Pickomino Game for an Gym Environment"""

from gymnasium.envs.registration import register

# Nur registrieren – Gymnasium lädt die Klasse erst bei gym.make()

register(
    id="Pickomino-v0",
    entry_point="pickomino_env.pickomino_gym_env:PickominoEnv",
    # kwargs={"num_players": 2},
    max_episode_steps=200,
)
