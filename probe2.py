# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 13:44:26 2025

@author: TAKO
"""

# import gym
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO  # WORKS GREAT! even better 8sec done!
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

# from stable_baselines3 import SAC # Soft Actor-Critic (SAC) is suitable for continuous action spaces
import pickomino_env

render_mode = None

# render_mode = "human"  # Set to None for no rendering, or "human" for rendering
env = Monitor(gym.make("Pickomino-v0"), "./logs/")
print(env)

# 	•	ent_coef fördert Diversität der Policy durch Entropie.
# 	•	target_kl begrenzt, wie weit die Policy pro Update vom alten Stand abweichen darf (trust-region-ähnlich).
algo = PPO("MultiInputPolicy", env, verbose=1, target_kl=0.02, ent_coef=0.02)
algo.learn(total_timesteps=200000)  # 1 step = one action! (not episode!)

# plotting
x, y = ts2xy(load_results("./logs/"), "timesteps")
plt.plot(x, y)
plt.xlabel("Timesteps")
plt.ylabel("Rewards")
plt.title("Learning Curve")
plt.grid()
plt.show()
