# Copilot Instructions for haft

## Project Overview
`haft` is a collection of small, standalone Python utilities, factored out so
they can be reused across projects rather than re-implemented. The organizing
principle is **each utility is its own submodule** under `haft/`, and the
top-level package itself contains almost nothing beyond re-exports. The goal is
to reduce code repetition and encourage reuse.

Currently the package contains a single submodule:

- **`haft.selection`** — a parser and resolver for a byte/Unicode range
  specification language. It parses strings like `0:10`, `0.5:1.0`, and `.100`
  into an `Offset`/`Range`/`Selection` hierarchy, resolves ranges to true byte
  offsets against raw content (using a smart `\r?\n` line rule), and extracts
  the corresponding text. Resolution and extraction are owned by `Selection`,
  with `Range` convenience wrappers; `encoding` is a resolve-time argument.

More submodules will be added over time. Keep each one self-contained and
independently importable. Cross-submodule dependencies are acceptable when they
serve reuse — that is the point of the package — but a submodule should not
reach outside `haft` for its core behavior.

## Architecture & Conventions

### Package Structure
- `haft/` — the package. Almost empty at the top level; real functionality
  lives in submodules.
- `haft/<submodule>.py` (or `haft/<submodule>/` for larger ones) — one utility
  per submodule, independently importable.
- `haft/__init__.py` — re-exports the stable public surface of each submodule.
- `tests/` — pytest-based test suite, one test module per submodule.
- `pyproject.toml` — package configuration; `setuptools` with
  `packages.find` (`include = ["haft*"]`, `exclude = ["tests*"]`).

### Submodule shape
- Each submodule declares its public API via `__all__` at module level.
- Public names are re-exported from `haft/__init__.py` as they stabilize.
- Prefer a small, method-based public surface; keep parsing/resolution logic
  cohesive within the submodule that owns it.

## Development Workflow

### Running Tests
```bash
pytest tests/
```

### Package Installation (Development Mode)
```bash
pip install -e '.[dev]'
```

## Python Standards
- Package version: `0.1.0` (update in `haft/__init__.py` and `pyproject.toml`
  together).
- Python 3.8+ compatibility required. Do **not** use PEP 585 builtin generics
  (`list[...]`, `tuple[...]`) or PEP 604 unions (`X | Y`) in runtime type
  annotations; use `typing.List`, `typing.Tuple`, `typing.Optional`.
- Use pytest for all tests, with descriptive docstrings.
- Follow PEP 8 and PEP 257 conventions.
- Type hint all functions and methods.
- Use spaces for indentation (4 spaces per level).

- NO WHITESPACE AT END OF LINES

- Naming Conventions:
  - Modules: `snake_case`
  - Classes: `PascalCase`
  - Functions/Methods: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`

- Single class per file should be preferred unless tightly coupled. Small
  support classes may be grouped if justified.
    - Large submodules should be organized into subdirectories with
      `__init__.py` files. Smaller file size is preferred for readability.

- Docstrings suggested for all public modules, classes, and methods.
    - Short functions (< 10 lines) should be well-named, have well-named
      arguments, and omit docstrings.

- Use named groups in regular expressions which need to extract multiple fields.

- Unexported helper methods should be prefixed with a single underscore.

- Free functions should be preferred over static class methods unless state
  management is required.

- Public APIs should be explicitly exported via `__all__` in `__init__.py` and
  at module level.

- Helper functions should be placed after the classes and functions which use
  them in the file.

- Nested try/except blocks should be avoided. Use helper functions to isolate
  error handling when necessary.
    - As few exception types as possible should be caught. Avoid bare except
      clauses.
    - Unless execution can meaningfully continue, exceptions should be allowed
      to propagate to the caller, and handled at the top level.
    - Exceptions should not be caught just to be re-raised.

- Prefer list, tuple, set, and dictionary comprehensions over for loops.

- Prefer loud failures over silent degradation: raise `ValueError` on invalid
  input rather than coercing to a default or silently returning empty/partial
  results.

## Critical Files
- `pyproject.toml` - Package configuration and dependencies.
