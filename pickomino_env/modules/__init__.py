```python
"""Needed for pylint import."""

def calculate_reward(worm_value, opponent_worm_value):
    """
    Calculate the reward for picking a tile.

    The reward is the sum of the worm value of the tile picked and the reduction in the opponent's worm value.

    Args:
        worm_value (int): The worm value of the tile picked.
        opponent_worm_value (int): The worm value of the tile in the opponent's stack.

    Returns:
        int: The total reward.
    """
    return worm_value + (opponent_worm_value // 2)  # Double reward for stealing
```