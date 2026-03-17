```python
## Rewards

The goal is to collect tiles in a stack. The winner is the player, which at the end of the game has the most worms
on her tiles. For the Reinforcement Learning Agent a reward equal to the value
(worms) of a tile is given when the tile is picked. For a failed attempt
(see rulebook), a corresponding negative reward is given. When a bot steals your
tile, a double reward is given, consisting of the value of the tile and the reduction in the opponent's score.

## Starting State

* `dice_collected` = [0, 0, 0, 0, 0, 0].
* `dice_rolled` = [3, 0, 1, 2, 0, 2] Random dice, sum = 8.
* `tiles_table` = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1].
* `tile_players` = [0, 0, 0] (with number_of_bots = 2).

## Episode End

The episode ends if one of the following occurs:

1. Termination: If there are no more tiles to take on the table = Game Over.
2. Termination: Action out of allowed range [0–5, 0-1].

### Truncation

Truncation: Attempt to break the rules, the game continues, and you have to give a new valid action.

### Failed Attempt

Note that a Failed Attempt means: If a tile is present, put it back on the table and get a negative reward.
However, the game continues, so the Episode does not end.

## Arguments

These must be specified.

| Parameter        | Type        | Default | Description                                                                                |
|------------------|-------------|---------|--------------------------------------------------------------------------------------------|
| `number_of_bots` | int         | 1       | Number of bot opponents (1-6) you want to play against                                     |
| `render_mode`    | str or None | None    | Visualization mode:<br/>None (training),<br/>"human" (display), or "rgb_array" (recording) |

## Setup

`pip install pickomino-env`

## Usage example

```python
import gymnasium as gym

# Create environment
env = gym.make("Pickomino-v0", render_mode="human", number_of_bots=2)

# Reset and get initial observation
obs, info = env.reset(seed=42)

# Run one episode
terminated = False
truncated = False
total_reward = 0

while not terminated and not truncated:
    # Agent selects action: (dice_face, roll_choice)
    action = env.action_space.sample()  # Random action for demo

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    # Double reward for stealing
    if reward > 0 and info['stolen']:
        reward *= 2
    total_reward += reward

    if truncated:
        print(f"Invalid action: {info['explanation']}")
        break

print(f"Episode finished. Total reward: {total_reward}")
env.close()
```