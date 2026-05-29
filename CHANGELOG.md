# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.4.1] - 2026-05-01

### Fixed

- Missing sprite files in package data causing FileNotFoundError on installation (#395)

## [1.4.0] - 2026-04-24

### Added

- `pickomino-play` console script entry point for manual play (#358)
- Game over screen in pygame GUI (#375)
- Play again option in game over screen (#380)
- Invalid action feedback in pygame GUI (#367)
- Bot count selection in GUI (#374)
- TROUBLESHOOTING.md (#384)
- TESTING.md (#387)
- SECURITY.md with bug bounty policy (#381)
- Link to Board Game Arena (#386)
- Bot play speed instructions to README (#385)
- Differences from the physical game section in the README (#389)
- Bot heuristic section in the README (#390)

### Changed

- README improvements: Episode End, Action Space, Observation Space, Info Dict, Setup/Installation (#382)
- Updated the GIF in the README to reflect the current GUI layout (#378)
- Render after each step instead of after the full turn (#361)
- Updated the render delay constant (#366)
- Removed command-line tool (CLI) dependency from manual play (#355)

### Fixed

- Permanent hover effect until the action is complete (#359)
- Foxtrot font is now readable, window size and button positions corrected (#360)
- Logging test fix in tests submodule (#325)

## [1.3.0] - 2026-03-09

### Added

- Image added to README (#303)
- Logging tests (#318)

### Changed

- CONTRIBUTING.md updated with merge restrictions (#301)
- The coverage threshold was changed to 92% (#308)
- Heuristic policy refactored to stay within complexity limits (#321)
- Image display centred and resized (#324)

### Fixed

- render_modes metadata and rgb_array window initialisation (#306)
- tiles.is_empty() always returns False (#312)
- Dead code in Renderer and PickominoEnv render signatures (#314)
- Failed attempt being overwritten (#323)
- Pre-commit config cleaned up after PyPI publishing (#317)
- pytest command in the workflow (#307)

## [1.2.0] - 2026-02-16

### Added

- Python 3.13 and 3.14 support (#248, #257)
- Logging for bot turns and actions (#290)
- Manual play with pygame GUI (#291)

### Changed

- Switched from pygame to pygame-ce (#254)
- Improved bot heuristic — prefers fewer dice when face value is equal (#268)
- Dropped Python 3.8 and 3.9 support (#257)

### Fixed

- Heuristic stop-condition bug where the bot would not stop on the same turn it first collects a worm (#286)

## [1.1.1] - 2026-01-19

### Fixed

- Version bumped from 1.0.7 to 1.1.1 for the PyPI release (#240)

## [1.1.0] - 2026-01-19

### Added

- macOS matrix, pytest, pyright to CI workflow (#225)
- Google-style docstrings (#235)
- Full package description for PyPI page (#238)
- Test timeouts for PPO tests (#223)

### Changed

- Renamed pickomino_gym_env.py to pickomino_env.py (#228)
- Refactored set_failed_no_tile_to_take into two functions (#209)

## [1.0.2] to [1.0.7] - 2026-01-14

### Fixed

- PyPI publish-workflow — multiple attempts to fix the automation pipeline (#211, #213, #214, #216, #218, #219, #220)

## [1.0.2] - 2026-01-14

### Added

- pygame rendering (#156, #161)
- Code of Conduct (#179)
- Farama project standards compliance (#169, #172)
- PyPI: automated the publish-workflow (#210)

### Changed

- Switched to Ruff for linting in pre-commit (#160, #168)
- Refactored to use Game class (#183, #186)
- Renamed RuleChecker class (#188)
- Validation returns converted to exceptions with descriptive messages (#197)
- Tests reorganised (#198)

### Fixed

- Tile 21 could be taken again (#157)
- Xenon complexity check for nested classes (#192)
- Submodule link (#194)

## [1.0.0] - 2025-11-19

### Added

- Initial public release

## [0.1.0] - 2025-08-21

### Added

- Initial pre-release at the end of the project
-
