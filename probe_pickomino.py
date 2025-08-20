import gymnasium as gym
import pickomino_env   # wichtig: löst das register() aus

env = gym.make("Pickomino-v0", num_players=2)
obs, info = env.reset()
print(obs)
