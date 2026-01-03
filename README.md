# panther-em

[![License](https://img.shields.io/pypi/l/panther-em.svg?color=green)](https://github.com/mgiammar/panther-em/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/panther-em.svg?color=green)](https://pypi.org/project/panther-em)
[![Python Version](https://img.shields.io/pypi/pyversions/panther-em.svg?color=green)](https://python.org)
[![CI](https://github.com/mgiammar/panther-em/actions/workflows/ci.yml/badge.svg)](https://github.com/mgiammar/panther-em/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mgiammar/panther-em/branch/main/graph/badge.svg)](https://codecov.io/gh/mgiammar/panther-em)

Pipelined AcceleratioN of Template matcHing via Eigendecomposition of Rotational projections in cryo-EM

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork mgiammar/panther-em --clone
# or just
# gh repo clone mgiammar/panther-em
cd panther-em
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
