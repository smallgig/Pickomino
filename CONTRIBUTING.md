# Contributing to Pickomino-Env

Contributions are welcome! We run two-week Sprints with issues assigned to contributors.
Feel free to comment on or open new issues. Please, label your issues.

## Creating Issues

- Use the issue template.
- Provide a clear description and reproduction steps (for bugs).
- Label the issue (type, size).
- Link related issues if applicable.
- Add criteria for when an issue is considered done.

## Branching Strategy

1. **Every change starts with an issue** Do not create a branch without an issue.
2. **Branch naming** Use issue number: `<issue-number>-<brief-description>`.
   You can do this automatically from GitHub.
    - Example: `246-change-to-pygame-ce-from-pygame`.
3. **Work on the branch** Never commit directly to `main`.
4. **Create Pull Request** from the branch. When a branch is merged to `main`
   the issue is automatically closed.
5. **Delete branch** After merge, delete your branch.

## Development Setup

We recommend **Python 3.14** for development (the latest supported version).
Any Python 3.10+ works locally. CI tests run on all supported versions.

```bash
git clone https://github.com/smallgig/Pickomino.git
cd Pickomino
pip install -e ".[dev]"
pre-commit install
```

## Development Workflow

### Definition of Ready

An issue is ready to be added to the next Sprint when:

- The issue is assigned to a developer.
- The issue size is estimated and has one and only one size label.
- The issue is labeled with a type (enhancement, bug, documentation, technical debt, test, continuous integration).
- The issue description contains criteria for when the issue is done.
- Developer agrees they understand what to do and have no open questions.

**Note:** If an issue is unclear (for example, size > M, architectural changes, blocks multiple issues),
create a pre-issue to analyze, plan, and write the real issue. The pre-issue is
done when the real issue is ready.

### Definition of Done

An issue is complete when:

- Criteria in the issue are fulfilled.
- Code changes are accompanied by updated comments, docstrings and README.
- Pull Request merged to main.
- Development branch deleted.
- Newly identified issues have been created (must not be ready).

## Before Submitting a Pull Request

- Run all pre-commit checks: `pre-commit run --all-files`
- Ensure tests pass: `pytest`
- Add tests for new features (maintain 95%+ coverage)

## Code Style

Ruff, black, pylint, pyright, mypy (strict mode)
Google-style docstrings
Type hints required.

## Questions?

Open an issue on GitHub with the label question.
