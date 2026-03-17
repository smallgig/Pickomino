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

```markdown
## Security & Bug Bounty

Found a bug? Valid reports are rewarded with a physical copy of the Pickomino board
game. See [SECURITY.md](SECURITY.md) for scope, timelines, and how to report.
```

```python
# SECURITY.md
# Security & Bug Bounty Policy

## Pickomino — Reinforcement Learning Environment

Pickomino is an open-source Reinforcement Learning environment for the board game
Pickomino/Heckmeck, confirmed as an external environment in
[Farama Gymnasium](https://gymnasium.farama.org/).

This is a small, part-time personal project. The bug bounty program is run in good
faith and on a best-effort basis. Please read this document carefully before reporting.

-----

## Reward

Valid bug reports are rewarded with **one physical copy of the Pickomino/Heckmeck board
game**, shipped via Amazon.de Prime to any destination covered by Amazon.de Prime
shipping.

> ⚠️ **One reward per unique bug.** The first valid report of a given issue wins.
> Duplicate reports are not eligible.

-----

## Scope

### In scope

- Incorrect implementation of Pickomino game rules
- Observation space or action space definition errors
- Reward function inaccuracies
- Non-compliance with the Farama Gymnasium API
- Dependency vulnerabilities (in `pyproject.toml` / `requirements`)
- Reproducibility or random seeding issues
- Packaging and release pipeline issues

### Out of scope

- The GitHub platform itself
- The Farama Gymnasium core library
- Feature requests framed as bugs
- Performance optimisation suggestions
- Theoretical or unproven issues without a reproducer
- Issues found by automated scanners run without prior permission

-----

## How to Report

### Security-sensitive issues

Use GitHub’s built-in private disclosure:
**Security → Report a vulnerability** (top of the repository page).

### Non-security bugs

Open a regular GitHub Issue and prefix the title with `[BUG BOUNTY]`.

### What to include

Please provide all of the following:

1. **A minimal Python reproducer** — a short script that demonstrates the issue
   against a specific tagged version of Pickomino.
1. **Expected behaviour** — what the correct output or behaviour should be, with a
   reference to the Pickomino/Heckmeck rulebook if applicable.
1. **Actual behaviour** — what the environment does instead.
1. **Environment details** — Python version, Gymnasium version, OS.

Reports without a working reproducer may not qualify for a reward.

-----

## Timelines

This is a part-time personal project. Please set expectations accordingly.

|Milestone                                    |Target               |
|---------------------------------------------|---------------------|
|Initial acknowledgement                      |14 days              |
|Triage decision (valid / invalid / duplicate)|30 days              |
|Fix or published workaround                  |90 days              |
|Public disclosure (security issues)          |After fix is released|

If a milestone is missed, feel free to send a polite follow-up on the issue or
advisory thread.

**Please do not disclose security-sensitive findings publicly before a fix is
released.**

-----

## Rules

- This program is not a legal contract. It operates entirely on mutual good faith.
- Rewards are sent once a fix is merged or a workaround is published.
- The maintainer reserves the right to adjust reward eligibility or close the program
  at any time, with notice posted in this file.
- By participating you agree not to exploit any vulnerability beyond what is strictly
  necessary to demonstrate it.
- The maintainer will credit all valid reporters by name (or handle) in the changelog
  and release notes, unless the reporter requests otherwise.

-----

## README Snippet

Add this to your `README.md`:

```markdown
## Security & Bug Bounty

Found a bug? Valid reports are rewarded with a physical copy of the Pickomino board
game. See [SECURITY.md](SECURITY.md) for scope, timelines, and how to report.
```

-----

*Bug Bounty v1.0 — March 2026*
```