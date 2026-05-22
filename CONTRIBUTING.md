# Contributing to MOUSE Environments

MOUSE is actively developed and contributions are very welcome — whether that's bug reports, new environments, wrappers, or documentation improvements.

## Ways to contribute

- **Bug reports** — open a GitHub issue with a minimal reproduction and the full error traceback.
- **Feature requests** — open an issue describing the use case. If you have a design idea, sketching it out in the issue first helps align before writing code.
- **Pull requests** — see the workflow below.
- **New environments** — if you add support for a new environment type or NS-Gym scheduler, sharing results as an issue or discussion is valuable.
- **Documentation** — edits to Markdown under `docs/` or the README are welcome (no doc site build step).

## Development setup

```bash
# Clone and create a virtual environment (Python 3.12, via uv)
git clone https://github.com/micahr234/mouse-env.git
cd mouse-env
source scripts/install.sh
```

This installs the package in editable mode with dev dependencies.

## Pull request workflow

1. Fork the repository and create a branch from `main`.
2. Make your changes. Keep commits focused — one logical change per commit.
3. Check for linter errors (`pyright src/`) before opening a PR.
4. Open a pull request against `main` with a clear description of what changed and why.

There are no formal tests yet — if you add a new feature, a short usage example in the PR description or in the relevant `docs/` page is sufficient.

## Code style

- Python 3.12+, type-annotated throughout.
- Follow the existing patterns: wrappers in `stack/`, env backends in `backends/`, config in `config.py`, factory in `factory.py`, public API in `__init__.py`, documentation in `docs/`.
- Avoid silent fallbacks — if a precondition isn't met, raise a clear error.
- Comments should explain *why*, not *what*.

## Releasing to PyPI

Publishing is automated by [`.github/workflows/publish.yml`](.github/workflows/publish.yml) using [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC). One-time setup on [pypi.org](https://pypi.org/manage/project/mouse-envs/settings/publishing/):

| Field | Value |
|-------|-------|
| Owner | `micahr234` |
| Repository | `mouse-env` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

To publish a release:

1. Bump `version` in `pyproject.toml` on `main`.
2. Commit, push, and create an annotated tag matching the version (e.g. `v0.1.1` for version `0.1.1`).
3. Push the tag: `git push origin v0.1.1` — the Publish workflow runs on tag push.

You can also run the workflow manually from the Actions tab (**Publish** → **Run workflow**) after the trusted publisher is configured.

## Questions

Open a GitHub Discussion or issue.
