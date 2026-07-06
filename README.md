# haft

`haft` is a small collection of standalone Python utilities, factored out so
they can be reused across projects instead of re-implemented. Each utility is
its own submodule under `haft/`; the top-level package itself holds almost
nothing.

Currently the package provides a single submodule:

- **`haft.selection`** — a parser and resolver for a compact byte/Unicode
  range specification language. It turns strings like `0:10`, `0.5:1.0`, or
  `.100` into true byte offsets against raw content, and extracts the
  corresponding text.

## Installation

Install the `haft` package using pip:

```bash
pip install -e .
```

Or install directly from the repository:

```bash
# Install from main branch
pip install git+https://github.com/bracket/haft.git@main

# Or install a specific version/tag
pip install git+https://github.com/bracket/haft.git@v0.1.0
```

For development (test dependencies):

```bash
pip install -e '.[dev]'
```

## `haft.selection`

The selection module specifies regions of content using a line-and-offset
grammar and resolves them to true byte offsets. Resolution operates on raw
content directly — line starts are found with a smart `\r?\n` newline rule, so
resolved offsets correspond exactly to `content[start:end]`.

### Grammar

```
selection = range1[, range2, ...]
range     = start_offset:end_offset | start_offset: | :end_offset | :
offset    = line_no[.[+-]?u?char_offset] | .[+-]?u?char_offset
```

- `line_no` is 0-indexed and may be negative (Python-style indexing).
- `char_offset` is a byte offset by default, or a Unicode character offset when
  prefixed with `u`.
- A `.N` offset with no line number is measured from the file start.
- `.12` is a true byte offset (encoding-independent); `.u12` is a Unicode
  character offset in the given encoding.
- A sign, if present, comes *before* the `u` flag (e.g. `1.-u10`). An explicit
  `+` is accepted and normalized away.
- Negative byte offsets are relative to the line/file start; negative Unicode
  offsets index from the end of the line.
- Resolved offsets are clamped to `[0, len(content)]`. Negative-length ranges
  are empty.

### Grammar examples

```
0:10        # first 10 lines
0:10,20:30  # lines 0-9 and 20-29
:10         # first 10 lines (from file start)
10:         # line 10 to end of file
0.5:1.0     # byte 5 of line 0 to byte 0 of line 1
0.u5:1.u0   # Unicode char 5 of line 0 to char 0 of line 1
1.-10       # 10 bytes before the start of line 1
1.-u10      # 10 Unicode chars from the end of line 1
.100        # byte 100 from the file start
```

### API

Resolution and extraction are owned by `Selection`; `Range` provides
single-range convenience wrappers. Line starts are computed once per
`Selection.resolve` call, so multi-range selections don't recompute them.

```python
from haft.selection import Selection, Range, Offset

content = b"hello\nworld\nfoo\n"

# Parse a selection string
sel = Selection.parse("0.0:1.3")

# Resolve to true (start, end) byte offsets, one tuple per range
sel.resolve(content)                 # -> [(0, 9)]

# Extract decoded text (ranges joined by `joiner`, default '\n')
sel.extract(content)                 # -> 'hello\nwor'

# Encoding is a resolve-time argument, not stored on the objects
sel.resolve(content, encoding="utf-8")
sel.extract(content, encoding="utf-8", joiner="")

# Range-level convenience: resolve returns a single unwrapped tuple
rng = Range.parse("0:1")[0]
rng.resolve(content)                 # -> (0, 6)
rng.extract(content)                 # -> 'hello\n'
```

Key contract: for any single range,
`extract == content[start:end].decode(encoding, errors='replace')`, where
`(start, end)` are the offsets returned by `resolve`.

## Development

Run the tests with pytest:

```bash
pytest tests/
```

## License

MIT

## Repository

https://github.com/bracket/haft
