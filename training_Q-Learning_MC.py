import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from random import random
from torch import optim
import pickomino_env  # wichtig: löst das register() aus

# Hyperparameters
episodes = 5000
gamma = 0.99  # discount factor: how much rewards down the line are worth
lr = 0.0001  # learning rate
use_epsilon = True  # True for epsilon-greedy exploration while LEARNING

env = gym.make("Pickomino-v0")  #  with visualization
save_file = "pickomino_test.pth"

print(f"Original observation space: {env.observation_space}")
print(f"Original action space: {env.action_space}")


def flatten_observation(obs_dict):
    """
    Manuelles Flattening der Dict-Observation
    """
    flattened_parts = []

    # dice_collected: shape (6,) -> 6 Werte
    flattened_parts.append(obs_dict["dice_collected"].flatten())

    # dice_rolled: shape (6,) -> 6 Werte
    flattened_parts.append(obs_dict["dice_rolled"].flatten())

    # tiles_table: Dict mit 16 Einträgen (i: 0 oder 1) -> 16 Werte
    tiles_values = []
    for i in range(21, 37):  # 21 bis 36 (16 Werte)
        if i in obs_dict["tiles_table"]:
            tiles_values.append(float(obs_dict["tiles_table"][i]))
        else:
            tiles_values.append(0.0)
    flattened_parts.append(np.array(tiles_values))

    # tile_player: 1 Wert
    flattened_parts.append(np.array([float(obs_dict["tile_player"])]))

    return np.concatenate(flattened_parts)


# Test the flattening
obs, info = env.reset()
flattened_obs = flatten_observation(obs)
observation_dimensions = len(flattened_obs)
print(f"Flattened observation dimensions: {observation_dimensions}")
print(f"Sample flattened obs shape: {flattened_obs.shape}")

# action_space ist MultiDiscrete(6, 2) - das bedeutet 2 actions mit je 6 möglichen Werten
# Das wird als 6*2 = 12 verschiedene Kombinationen behandeln
action_dimensions = np.prod(env.action_space.nvec)  # 6 * 2 = 12
print(f"Action dimensions: {action_dimensions}")

q_values = nn.Sequential(
    nn.Linear(observation_dimensions, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, action_dimensions),
)

optimizer = optim.Adam(q_values.parameters(), lr=lr)

try:
    checkpoint = torch.load(save_file)
    q_values.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print("💡 pre-trained model loaded")
except:
    print("⚠️ No pre-trained model found, starting from scratch.")


def action_to_index(action):
    """Konvertiert MultiDiscrete action zu einem einzelnen Index"""
    return action[0] * env.action_space.nvec[1] + action[1]


def index_to_action(index):
    """Konvertiert einen Index zurück zu MultiDiscrete action"""
    action_0 = index // env.action_space.nvec[1]
    action_1 = index % env.action_space.nvec[1]
    return np.array([action_0, action_1])


def policy_random(state):
    """Random policy - sampelt eine zufällige Aktion"""
    action = env.action_space.sample()
    return action


def policy_q(state):
    """Q-Learning policy - wählt die beste Aktion basierend auf Q-Werten"""
    with torch.no_grad():  # Keine Gradienten für Inferenz
        action_values = q_values(state)  # vector of q-values
        action_index = torch.argmax(action_values).item()  # Get the index of the max value
    return index_to_action(action_index)


def policy(state, epsilon):
    """Epsilon-greedy policy"""
    if random() < epsilon:
        return policy_random(state)
    else:
        return policy_q(state)


# Liste für das Tracking der Gewinne
total_gains = []

for step in range(episodes):
    if use_epsilon:
        epsilon = max(0.1, 1 - step / (episodes * 0.8))  # linear decay with minimum exploration
    else:
        epsilon = 0  # no exploration, only exploitation, deterministic policy

    obs_dict, info = env.reset()
    obs_flat = flatten_observation(obs_dict)
    state = torch.tensor(obs_flat, dtype=torch.float32)  # Konvertiere zu Tensor

    done = False
    episode = []
    total_gain = 0
    step_count = 0
    max_steps = 1000  # Verhindere unendliche Episoden

    while not done and step_count < max_steps:
        action = policy(state, epsilon)

        try:
            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            total_gain += reward

            # Konvertiere action zu index für das Speichern
            action_index = action_to_index(action)

            next_obs_flat = flatten_observation(next_obs_dict)
            next_state = torch.tensor(next_obs_flat, dtype=torch.float32)

            episode.append((state, action_index, reward))
            state = next_state
            done = terminated or truncated
            step_count += 1

        except Exception as e:
            print(f"Error during step: {e}")
            print(f"Action was: {action}")
            print(f"Action type: {type(action)}")
            break

    # Monte Carlo update learning - nur wenn Episode Daten hat
    if len(episode) > 0:
        Gain = 0
        for state_ep, action_ep, reward_ep in reversed(episode):
            Gain = reward_ep + gamma * Gain
            prediction = q_values(state_ep)[action_ep]
            # Train the model
            loss = (prediction - Gain) ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    total_gains.append(total_gain)

    if step % 100 == 0:
        avg_gain = np.mean(total_gains[-100:]) if len(total_gains) >= 100 else np.mean(total_gains)
        print(f"Episode {step}  gain: {total_gain:.2f}  avg_gain_100: {avg_gain:.2f}  epsilon: {epsilon:.3f}")
        torch.save({"model_state": q_values.state_dict(), "optimizer_state": optimizer.state_dict()}, save_file)

print("Training completed!")

# plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# Plot 1: Alle Gewinne
plt.subplot(2, 2, 1)
plt.plot(total_gains)
plt.title("Total Gain per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Gain")
plt.grid(True)

# Plot 2: Moving average
window_size = 100
if len(total_gains) > window_size:
    moving_avg = np.convolve(total_gains, np.ones(window_size) / window_size, mode="valid")
    plt.subplot(2, 2, 2)
    plt.plot(moving_avg)
    plt.title(f"Moving Average Total Gain (window size: {window_size})")
    plt.xlabel("Episode")
    plt.ylabel("Average Total Gain")
    plt.grid(True)

# Plot 3: Letzten 1000 Episoden
plt.subplot(2, 2, 3)
recent_gains = total_gains[-1000:] if len(total_gains) > 1000 else total_gains
plt.plot(recent_gains)
plt.title("Recent Episodes Gain")
plt.xlabel("Episode")
plt.ylabel("Total Gain")
plt.grid(True)

# Plot 4: Histogramm der Gewinne
plt.subplot(2, 2, 4)
plt.hist(total_gains, bins=50, alpha=0.7)
plt.title("Distribution of Total Gains")
plt.xlabel("Total Gain")
plt.ylabel("Frequency")
plt.grid(True)

plt.tight_layout()
plt.show()
