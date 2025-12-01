## Description

An environment conforming to the **Gymnasium** API for the dice game **Pickomino (Heckmeck am Bratwurmeck)**
Goal: train a Reinforcement Learning agent for optimal play. That is, decide, which face of the dice to collect,
when to roll and when to stop.

## Action Space

The Action space is a tuple with two integers.
Tuple(int, int)

Action = [dice_face (1-6), action_type (0=roll, 1=stop)].

- 1-6: Face of the dice, which you want to take.
- 0-1: Roll or stop.

## Observation Space

The observation is a `dict` with shape `(4,)` with the values corresponding to the following: dice, table and player.

| Observation    | Min | Max | Shape              |
|----------------|-----|-----|--------------------|
| dice_collected | 0   | 8   | (6,)               |
| dice_rolled    | 0   | 8   | (6,)               |
| tiles_table    | 0   | 1   | (16,)              |
| tile_player    | 0   | 36  | number_of_bots + 1 |

**Note:** There are eight dice to roll and collect. A die has six sides with the number of eyes one through
five, but a worm instead of a six.
The values correspond to the number of eyes, with the worm also having the value five (and not six!).
The 16 tiles are numbered 21 to 36 and have worm values from one to four in spread in four groups.
The game is for 2 to 7 players. Here your Reinforcement Learning Agent is the first player. The
other players are computer bots.
The bots play, according to a heuristic. When you create the environment,
you have to define the number of bots.

For a more detailed description of the rules, see the file pickomino-rulebook.pdf.
You can play the game online here: https://www.maartenpoirot.com/pickomino/.
The heuristic used by the bots is described here: https://frozenfractal.com/blog/2015/5/3/how-to-win-at-pickomino/.

## Rewards

The goal is to collect tiles in a stack. The winner is the player which at the end of the game has the most worms
on her tiles. For the Reinforcement Learning Agent a reward equal to the value
(worms) of a tile is given when the tile is picked. In case of a failed attempt
(see rulebook), a corresponding negative reward is given. When a bot steals your
tile, no negative reward is given. Hence, the total reward at the end of the game
can be greater than the score.

## Starting State

* dice_collected = [0, 0, 0, 0, 0, 0]
* dice_rolled = [3, 0, 1, 2, 0, 2] Random dice, sum = 8
* tiles_table = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
* tile_player = [0, 0, 0] number_of_bots = 2

## Episode End

The episode ends if any one of the following occurs:

1. Termination: If the table is empty = Game Over.
2. Termination: Action out of allowed range.
3. Truncation: Attempt to break rules, the game continues, and you have to give a new valid action.
4. Failed Attempt: If a tile is present, put it back on the table and get a negative reward.

## Arguments

For visualisation set render_mode = "human", None for training.

When you create the environment, you have to specify the number of bots you want to play
against (1 to 6)

## Setup

`pip install pickomino-env`

## Usage example

```python
import gymnasium as gym
env = gym.make("Pickomino-v0", render_mode=None, number_of_bots=2)
```
