# Contributing to MOUSE Environments

MOUSE is actively developed and contributions are very welcome — whether that's bug reports, new environments, wrappers, or documentation improvements.

## Ways to contribute

- **Bug reports** — open a GitHub issue with a minimal reproduction and the full error traceback.
- **Feature requests** — open an issue describing the use case. If you have a design idea, sketching it out in the issue first helps align before writing code.
- **Pull requests** — see the workflow below.
- **New environments** — if you add support for a new environment type, sharing results as an issue or discussion is valuable.
- **Documentation** — edits to Markdown under `docs/` or the README are welcome (no doc site build step).

## Development setup

```bash
# Clone and create a virtual environment (Python 3.12, via uv)
git clone https://github.com/micahr234/mouse-env.git
cd mouse-env
source scripts/install.sh
```

This installs the package in editable mode with dev dependencies (including Jupyter for [`examples/`](examples/) notebooks).

Notebooks under [`examples/`](examples/) are committed **without** outputs. The repo's workspace settings set `"notebook.transientOutputs": true` in [`.vscode/settings.json`](.vscode/settings.json), so Cursor/VS Code never writes cell outputs to disk when you run and save a notebook here.

If you edit notebooks with a different tool (browser Jupyter, `nbconvert`, scripts), that setting does not apply — clear outputs before committing, e.g. `jupyter nbconvert --clear-output --inplace examples/*.ipynb`.

## Pull request workflow

1. Fork the repository and create a branch from `main`.
2. Make your changes. Keep commits focused — one logical change per commit.
3. Run tests (`.venv/bin/pytest`) and check for linter errors (`pyright src/`) before opening a PR.
4. Open a pull request against `main` with a clear description of what changed and why.

Tests live under [`tests/`](tests/):

- `test_smoke.py` — core env configs (CartPole, tabular envs, custom `env_fn`, reward shaping)
- `test_q_star.py` — expert Q* adapters offline (local SB3 checkpoint, tabular pickle; no Hugging Face)
- `test_atari.py` — ALE vector env + preprocessing via `env_fn` (requires the `atari` extra / `ale_py` ROMs)
- `test_ns_gym.py` — non-stationary env via `env_fn` (requires the `non-stationary` extra / `ns_gym`)

If you add a new feature, add or extend a test under [`tests/`](tests/) and/or a notebook under [`examples/`](examples/).

## Code style

- Python 3.12+, type-annotated throughout.
- Follow the existing patterns: config in `config.py`, build in `build.py`, wrappers in `wrappers.py`, formatting in `format.py`, first-party worlds in `worlds/`, expert Q* plumbing and MDP solvers in `experts/`, public API in `__init__.py`. Third-party envs (Atari, NS-Gym) are built by users via `env_fn` rather than bundled integrations. User-facing docs live in [`docs/guide.md`](docs/guide.md); implementation details belong in code comments and docstrings.
- Avoid silent fallbacks — if a precondition isn't met, raise a clear error.
- Comments should explain *why*, not *what*.

## Releasing to PyPI

Publishing is automated by [`.github/workflows/publish.yml`](.github/workflows/publish.yml) using [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC).

### Publishing a version

To publish a release:

1. Bump `version` in `pyproject.toml` on `main`.
2. Commit, push, and create an annotated tag matching the version (e.g. `v0.1.1` for version `0.1.1`).
3. Push the tag: `git push origin v0.1.1` — the Publish workflow runs on tag push.

You can also run the workflow manually from the Actions tab (**Publish** → **Run workflow**).

## Questions

Open a GitHub Discussion or issue.
