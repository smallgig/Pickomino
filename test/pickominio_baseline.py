# import gym
import gymnasium as gym

from stable_baselines3 import PPO  # WORKS GREAT! even better 8sec done!
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt

import time

start_time = time.time()
log_dir = "../logs/"

render_mode = None
# render_mode = "human"  # Set to None for no rendering, or "human" for rendering
# env = Monitor(gym.make("Pickomino-v0"), log_dir)
env = gym.make("Pickomino-v0")
algo = PPO("MultiInputPolicy", env)
algo.learn(total_timesteps=100000)  # 1 step = one action! (not episode!)


# plotting
x, y = ts2xy(load_results(log_dir), "timesteps")
plt.plot(x, y)
plt.xlabel("Timesteps")
plt.ylabel("Rewards")
plt.title("Learning Curve")
plt.grid()
plt.show()


end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")
