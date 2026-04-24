# Testing

## Framework
[pytest](https://pytest.org/) 8.x (pytest 9 is not supported).

## Test Suite
The test suite is maintained in a private submodule. Contributors requiring
access should contact the maintainer.

## Running Tests
Tests are run automatically via GitHub Actions on every push and pull request.
PRs from forks will trigger CI, but the run may fail because secrets needed to
access the private `tests` submodule are not available to forked workflows.
Contact the maintainer if you need help validating changes against the full test suite.

## Pre-commit
A subset of checks runs via pre-commit hooks on every commit. Tests also run
via pre-commit, but at the `pre-push` stage rather than on every commit.

## Coverage
95%+ test coverage is maintained and enforced in CI.

---

## Test Categories

### Fast Tests
- Location: `tests/`
- Purpose: Smoke tests — catch regressions quickly.
- Assert basic invariants, input validation, no crashes.

### Rules Tests
- Location: `tests/`
- Purpose: Verify Pickomino game rules are correctly implemented.
- Each test name describes the rule it verifies.
- Example: `test_bust_returns_top_tile_and_removes_highest_from_table`

---

## Naming Convention

```
test_<rule_or_behavior_described_in_plain_english>

Good: test_bot_stops_when_collecting_worm_reaches_tile
Bad:  test_step_valid_action
```
