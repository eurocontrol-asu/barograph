# Contributing to barograph

barograph is a uv workspace. Run the commands below from its root unless a
package directory is given. Python 3.12+, uv and Git are required.

## Development setup

```bash
git clone https://github.com/eurocontrol-asu/barograph.git
cd barograph
uv sync --all-packages --all-groups
uv run prek install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
```

The sync installs every member with its optional backends (GRIB decoding,
pandas, Open-Meteo reader) and the development, documentation and quality
groups.

## Making changes

1. Create a branch and scope the change to one package or to workspace files.
2. Read that package's public contracts and tests.
3. Run the package tests and `make lint`.
4. Build the documentation when examples, APIs, navigation or prose change.
5. Commit with a conventional message, e.g. `fix(met): flag points below the surface`.
6. Open a pull request describing the behaviour change, how it was validated
   and its remaining limits.

Each package's `pyproject.toml` owns its test and typing configuration.

## Tests and checks

```bash
uv run --package barograph-met --directory packages/barograph-met pytest   # one package
make test          # every package
make lint          # ruff + format check + mypy per package
make quality       # axm audit + axm init_check
```

Tests follow a three-level pyramid in `tests_<package>/`:

- `unit/`: no real I/O (in-memory stores and fake sources), mirrors `src/`;
- `integration/`: real I/O on local synthetic stores and mirrors;
- `e2e/`: separate processes.

Tests never touch the network. Checks against the public buckets are marked
`live` and run only on demand: `pytest -m live`.

## Documentation

```bash
make docs-serve    # preview
make docs-build    # strict build, as in CI
```

Pages follow the Diátaxis split (tutorials, how-to guides, reference,
explanation); the API reference is generated from docstrings.

## Releases

Each package is released on its own long tag, e.g. `barograph-met/v0.1.0`:
the tag triggers the tests, the PyPI upload (Trusted Publishing) and a GitHub
release with a git-cliff changelog.

## Licence

By contributing you agree that your contributions are licensed under the
EUPL-1.2 (see `LICENSE`).
