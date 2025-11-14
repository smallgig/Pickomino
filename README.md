
## Description

An environment conforming to the **Gymnasium** API for the dice game **Pickomino (Heckmeck am Bratwurmeck)**
Goal: train a Reinforcement Learning agent for optimal play (which dice to collect, when to stop).

## Action Space

The Action space is a tuple with two integers.
Tuple(int, int)
- 1-6: Dice which u want to take
- 0-1: Roll or stop

**Note**:

## Observation Space

The observation is a `dict` with shape `(4,)` with the values corresponding to the following dice, table and player:

| Num | Observation    | Min | Max | Shape             |
|-----|----------------|----|-----|--------------------|
| 0   | dice_collected | 0  | 8   | 6                  |
| 1   | dice_rolled    | 0  | 8   | 6                  |
| 2   | tiles_table    | 0  | 1   | 16                 |
| 3   | tile_player    | 0  | 36  | number_of_bots + 1 |

**Note:** While the ranges above denote the possible values for observation space of each element,
    it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
-  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
   if the cart leaves the `(-2.4, 2.4)` range.
-  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
   if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

## Rewards
Since the goal is to keep the pole upright for as long as possible, by default, a reward of `+1` is given for every step taken, including the termination step. The default reward threshold is 500 for v1 and 200 for v0 due to the time limit on the environment.

If `sutton_barto_reward=True`, then a reward of `0` is awarded for every non-terminating step and `-1` for the terminating step. As a result, the reward threshold is 0 for v0 and v1.

## Starting State
All observations are assigned a uniformly random value in
* dice_collected = [0, 0, 0, 0, 0, 0]
* dice_rolled = [3, 0, 1, 2, 0, 2] Random dice, sum = 8
* tiles_table = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
* tile_player = [0, 0, 0] number_of_bots = 2

## Episode End
The episode ends if any one of the following occurs:

1. Termination: If the table is empty = Game Over
2. Termination: Action out of allowed range
3. Truncation: Attempt to break rules, game continues
4. Failed Attempt: If tile is present put it back on table and get negative reward

## Arguments

Cartpole only has `render_mode` as a keyword for `gymnasium.make`.
On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

| Parameter               | Type       | Default                 | Description                                                                                   |
|-------------------------|------------|-------------------------|-----------------------------------------------------------------------------------------------|
| `sutton_barto_reward`   | **bool**   | `False`                 | If `True` the reward function matches the original sutton barto implementation                |

## Vectorized environment
