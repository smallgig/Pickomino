```python
"""Example: Train a Reinforcement Learning Agent on Pickomino environment."""

import gymnasium as gym
import pickomino_env  # noqa: F401 Triggers registration

# Create a Pickomino environment with 2 bots
env = gym.make("Pickomino-v0", render_mode=None, number_of_bots=2)
obs, info = env.reset(seed=42)

# Train the agent for 100 steps
for step in range(100):
    # Sample an action from the environment's action space
    action = env.action_space.sample()
    
    # Perform the action and get the new observation, reward, and other information
    obs, reward, terminated, truncated, info = env.step(action)

    # Double the reward for stealing by adding the opponent's worm value reduction
    if 'steal' in info and info['steal']:
        reward += info['opponent_worms']

    # Reset the environment if the current step is terminated
    if terminated:
        obs, info = env.reset()

# Close the environment
env.close()
```