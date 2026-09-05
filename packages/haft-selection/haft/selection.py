"""Selection parser and resolver for fine-grained byte range specification.

This module provides a parser and class hierarchy for specifying byte or
Unicode character ranges within content using a line-and-offset notation,
along with methods to resolve those ranges to true byte offsets and extract
the corresponding content.

Grammar::

    selection = range1[, range2, ...]
    range = (start_offset:end_offset) | (start_offset:) | (:end_offset)
    offset = (line_no(.[+-]?u?char_offset?)) | (.[+-]?u?char_offset)

Where:

- ``line_no`` is 0-indexed and can be negative (Python-style)
- ``char_offset`` is a byte or Unicode character offset from line or file start
- Negative byte char_offsets are relative to the line/file start (they may
  resolve before the start; the final position is clamped to ``[0, total]``)
- Negative Unicode (``u``) char_offsets index from the *end* of the line
- An explicit ``+`` sign on char_offset is accepted but normalised away
- ``u`` flag indicates Unicode character offsets
- The sign (if any) appears *before* the ``u`` flag
- Empty start means file beginning, empty end means file end
- Negative length ranges are empty; zero-length represents a cursor position

Examples::

    0:10        # First 10 lines
    0:10,20:30  # Lines 0-9 and 20-29
    :10         # First 10 lines (from file beginning)
    10:         # Line 10 to file end
    0.5:1.0     # Byte 5 of line 0 to byte 0 of line 1
    0.u5:1.u0   # Unicode character 5 of line 0 to character 0 of line 1
    1.-10       # 10 bytes before the start of line 1
    1.-u10      # 10 Unicode chars from the end of line 1
    1.+10       # Same as 1.10 (explicit positive sign)
"""

import re
from typing import List, Optional, Tuple

__all__ = ['Offset', 'Range', 'Selection']


class Offset:
    """Represents a position in a file as a line number and character offset.

    An offset specifies a position using an optional line number and an
    optional character offset within that line (or from the file start if
    no line number is given).

    Attributes:
        line_no: 0-indexed line number, or None for a file-level offset.
            Negative values use Python-style indexing from the end.
        char_offset: Byte or Unicode character offset from the line (or
            file) start. None if no character offset is specified.
        is_unicode: If True, ``char_offset`` counts Unicode characters
            rather than raw bytes. Default is False.
    """

    def __init__(
        self,
        line_no: Optional[int] = None,
        char_offset: Optional[int] = None,
        is_unicode: bool = False,
    ) -> None:
        if line_no is None and char_offset is None:
            raise ValueError(
                "Offset must specify at least a line number or a character offset"
            )
        self.line_no = line_no
        self.char_offset = char_offset
        self.is_unicode = is_unicode

    @classmethod
    def parse(cls, s: str, pos: int = 0) -> Tuple['Offset', int]:
        """Parse an offset from a string at a given position.

        Recognises the following forms::

            line_no                   e.g. 0, -1, 42
            line_no.char_offset       e.g. 0.5, -1.0
            line_no.+char_offset      e.g. 0.+5 (same as 0.5)
            line_no.-char_offset      e.g. 0.-5 (negative offset)
            line_no.uchar_offset      e.g. 0.u5, 2.u10
            line_no.+uchar_offset     e.g. 0.+u5 (same as 0.u5)
            line_no.-uchar_offset     e.g. 0.-u5 (negative Unicode offset)
            .char_offset              e.g. .100  (file-level byte offset)
            .+char_offset             e.g. .+100 (same as .100)
            .-char_offset             e.g. .-100 (negative file-level byte offset)
            .uchar_offset             e.g. .u100 (file-level Unicode offset)
            .-uchar_offset            e.g. .-u100 (negative file-level Unicode offset)

        Args:
            s: The string to parse.
            pos: The starting position within the string.

        Returns:
            A tuple of ``(Offset, new_position)`` where ``new_position``
            is the index of the first character not consumed.

        Raises:
            ValueError: If the string does not contain a valid offset at
                ``pos``.
        """
        if pos >= len(s):
            raise ValueError(
                f"Expected offset at position {pos}, got end of string"
            )

        line_no: Optional[int] = None
        char_offset: Optional[int] = None
        unicode_flag = False

        if s[pos].isdigit() or s[pos] == '-':
            line_no, pos = _parse_int(s, pos)
            if pos < len(s) and s[pos] == '.':
                pos += 1  # consume '.'
                if _can_start_char_offset(s, pos):
                    char_offset, unicode_flag, pos = _parse_char_offset(s, pos)
        elif s[pos] == '.':
            pos += 1  # consume '.'
            if not _can_start_char_offset(s, pos):
                raise ValueError(
                    f"Expected digits after '.' at position {pos}"
                )
            char_offset, unicode_flag, pos = _parse_char_offset(s, pos)
        else:
            raise ValueError(
                f"Expected digit, '-', or '.' at position {pos}, "
                f"got {s[pos]!r}"
            )

        return cls(line_no=line_no, char_offset=char_offset, is_unicode=unicode_flag), pos

    def __str__(self) -> str:
        """Return the canonical string representation of this offset.

        Positive char_offset values are emitted without a sign.  Negative
        values include a ``-`` sign placed before the ``u`` flag, e.g.
        ``1.-u10``.  The ``+`` sign is never emitted.
        """
        if self.char_offset is not None:
            u_flag = 'u' if self.is_unicode else ''
            sign = '-' if self.char_offset < 0 else ''
            abs_offset = abs(self.char_offset)
            char_str = f".{sign}{u_flag}{abs_offset}"
        else:
            char_str = ''
        if self.line_no is not None:
            return f"{self.line_no}{char_str}"
        return char_str

    def __repr__(self) -> str:
        return (
            f"Offset(line_no={self.line_no!r}, "
            f"char_offset={self.char_offset!r}, "
            f"is_unicode={self.is_unicode!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Offset):
            return NotImplemented
        return (
            self.line_no == other.line_no
            and self.char_offset == other.char_offset
            and self.is_unicode == other.is_unicode
        )


class Range:
    """Represents a half-open interval between two offsets in a file.

    A range specifies a region from ``start`` (inclusive) to ``end``
    (exclusive).  Either endpoint may be ``None``, indicating the file
    beginning or file end respectively.

    Ranges with a negative length (i.e. ``start`` sorts after ``end``)
    are considered empty.  A range where both endpoints are equal
    represents a cursor position.

    Attributes:
        start: The start offset (inclusive), or ``None`` for the file
            beginning.
        end: The end offset (exclusive), or ``None`` for the file end.
    """

    def __init__(
        self,
        start: Optional[Offset],
        end: Optional[Offset],
    ) -> None:
        self.start = start
        self.end = end

    @classmethod
    def parse(cls, s: str, pos: int = 0) -> Tuple['Range', int]:
        """Parse a range from a string at a given position.

        Recognises the following forms::

            start:end    e.g. 0:10
            start:       e.g. 10:
            :end         e.g. :10
            :            (entire file)

        Args:
            s: The string to parse.
            pos: The starting position within the string.

        Returns:
            A tuple of ``(Range, new_position)``.

        Raises:
            ValueError: If the string does not contain a valid range at
                ``pos``.
        """
        start: Optional[Offset] = None
        end: Optional[Offset] = None

        if pos < len(s) and s[pos] != ':':
            start, pos = Offset.parse(s, pos)

        if pos >= len(s) or s[pos] != ':':
            raise ValueError(
                f"Expected ':' at position {pos} in range"
            )
        pos += 1  # consume ':'

        if _can_start_offset(s, pos):
            end, pos = Offset.parse(s, pos)
        elif pos < len(s) and s[pos] != ',':
            raise ValueError(
                f"Unexpected character {s[pos]!r} at position {pos} in range"
            )

        return cls(start=start, end=end), pos

    def resolve(self, content: bytes, *, encoding: str = 'utf-8') -> Tuple[int, int]:
        """Resolve this range to a ``(start, end)`` byte offset pair.

        Convenience wrapper over ``Selection([self]).resolve``.

        Args:
            content: Raw file content as bytes.
            encoding: Character encoding for Unicode offset calculations.

        Returns:
            A ``(start, end)`` tuple of true byte offsets into *content*.
            The start is clamped to ``[0, len(content)]``; negative-length
            ranges return ``(start, start)`` (empty).
        """
        pairs = Selection([self]).resolve(content, encoding=encoding)
        return pairs[0]

    def extract(self, content: bytes, *, encoding: str = 'utf-8') -> str:
        """Extract the content for this range.

        Convenience wrapper over ``Selection([self]).extract``.

        Args:
            content: Raw file content as bytes.
            encoding: Character encoding for decoding and Unicode offsets.

        Returns:
            Decoded string ``content[start:end].decode(encoding, errors='replace')``.
            Returns an empty string for negative-length ranges.
        """
        return Selection([self]).extract(content, encoding=encoding)

    def __str__(self) -> str:
        """Return the canonical string representation of this range."""
        start_str = str(self.start) if self.start is not None else ''
        end_str = str(self.end) if self.end is not None else ''
        return f"{start_str}:{end_str}"

    def __repr__(self) -> str:
        return f"Range(start={self.start!r}, end={self.end!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Range):
            return NotImplemented
        return self.start == other.start and self.end == other.end


class Selection:
    """Represents a collection of file ranges forming a named selection.

    A selection is a comma-separated list of :class:`Range` objects that
    together specify which parts of a file are included.

    Attributes:
        ranges: Ordered list of :class:`Range` objects making up the
            selection.
    """

    def __init__(self, ranges: List[Range]) -> None:
        self.ranges = ranges

    @classmethod
    def parse(cls, s: str) -> 'Selection':
        """Parse a complete selection string.

        Args:
            s: The selection string to parse.

        Returns:
            A :class:`Selection` object.

        Raises:
            ValueError: If the string is empty or does not contain a valid
                selection.
        """
        s = s.strip()
        if not s:
            raise ValueError("Selection string is empty")

        ranges: List[Range] = []
        pos = 0
        while pos < len(s):
            r, pos = Range.parse(s, pos)
            ranges.append(r)
            if pos < len(s):
                if s[pos] == ',':
                    pos += 1  # consume ','
                    while pos < len(s) and s[pos] == ' ':
                        pos += 1  # skip optional spaces after comma
                    if pos >= len(s):
                        raise ValueError("Trailing comma in selection string")
                else:
                    raise ValueError(
                        f"Expected ',' or end of string at position {pos}, "
                        f"got {s[pos]!r}"
                    )

        return cls(ranges=ranges)

    def resolve(self, content: bytes, *, encoding: str = 'utf-8') -> List[Tuple[int, int]]:
        """Resolve all ranges to true ``(start, end)`` byte offset pairs.

        Line starts are computed once from the raw content using ``\\r?\\n``
        as the line separator.  Resolved offsets are clamped to
        ``[0, len(content)]``.  Negative-length ranges produce
        ``(start, start)`` (i.e. empty).

        Args:
            content: Raw file content as bytes.
            encoding: Character encoding for Unicode offset calculations.

        Returns:
            A list of ``(start, end)`` tuples, one per range, in order.
        """
        line_starts = _build_line_starts(content)
        total = len(content)
        n_logical = _count_logical_lines(line_starts, total)

        result = []
        for r in self.ranges:
            start = _resolve_offset(
                r.start, content, line_starts, n_logical, total, encoding,
                is_start=True,
            )
            end = _resolve_offset(
                r.end, content, line_starts, n_logical, total, encoding,
                is_start=False,
            )
            start = max(0, min(start, total))
            end = max(0, min(end, total))
            if end < start:
                end = start
            result.append((start, end))
        return result

    def extract(
        self,
        content: bytes,
        *,
        encoding: str = 'utf-8',
        joiner: str = '\n',
    ) -> str:
        """Extract and join the content for all ranges.

        Args:
            content: Raw file content as bytes.
            encoding: Character encoding for decoding and Unicode offsets.
            joiner: String used to join multiple range extracts.
                Defaults to ``'\\n'``.

        Returns:
            The extracted ranges decoded and joined by *joiner*.
            Each range's bytes are decoded with
            ``errors='replace'``.  Negative-length ranges contribute
            an empty string.
        """
        pairs = self.resolve(content, encoding=encoding)
        parts = [
            content[start:end].decode(encoding, errors='replace')
            for start, end in pairs
        ]
        return joiner.join(parts)

    def __str__(self) -> str:
        """Return the canonical string representation of this selection."""
        return ','.join(str(r) for r in self.ranges)

    def __repr__(self) -> str:
        return f"Selection(ranges={self.ranges!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Selection):
            return NotImplemented
        return self.ranges == other.ranges


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_int(s: str, pos: int) -> Tuple[int, int]:
    """Parse a signed integer from *s* starting at *pos*."""
    start = pos
    if pos < len(s) and s[pos] == '-':
        pos += 1
    if pos >= len(s) or not s[pos].isdigit():
        raise ValueError(f"Expected digit at position {pos}")
    while pos < len(s) and s[pos].isdigit():
        pos += 1
    return int(s[start:pos]), pos


def _is_unicode_flag_at(s: str, pos: int) -> bool:
    """Return True if a ``u`` unicode flag starts at *pos*.

    The ``u`` character is only treated as a flag when it is immediately
    followed by a digit.
    """
    return pos < len(s) and s[pos] == 'u' and pos + 1 < len(s) and s[pos + 1].isdigit()


def _parse_char_offset(s: str, pos: int) -> Tuple[int, bool, int]:
    """Parse a signed character offset with an optional unicode flag.

    The expected form is ``[+-]?u?digits``.  The sign (if present) comes
    *before* the ``u`` flag.  A ``+`` sign is accepted but normalised: the
    returned value is always a plain signed integer.

    Args:
        s: The string to parse.
        pos: Starting position within *s*.

    Returns:
        A tuple ``(value, is_unicode, new_pos)``.

    Raises:
        ValueError: If no digits follow the optional sign/flag.
    """
    has_sign = pos < len(s) and s[pos] in ('+', '-')
    sign = 1
    if has_sign:
        if s[pos] == '-':
            sign = -1
        pos += 1

    is_unicode = False
    if _is_unicode_flag_at(s, pos):
        is_unicode = True
        pos += 1

    if pos >= len(s) or not s[pos].isdigit():
        if has_sign and not is_unicode and pos < len(s) and s[pos] == 'u':
            raise ValueError(
                f"Expected digit after 'u' flag at position {pos + 1}"
            )
        if has_sign:
            raise ValueError(
                f"Expected digit or 'u' flag after sign at position {pos}"
            )
        raise ValueError(f"Expected digit at position {pos}")

    start = pos
    while pos < len(s) and s[pos].isdigit():
        pos += 1

    return sign * int(s[start:pos]), is_unicode, pos


def _can_start_char_offset(s: str, pos: int) -> bool:
    """Return True if the character at *pos* could begin a char offset."""
    if pos >= len(s):
        return False
    c = s[pos]
    if c.isdigit():
        return True
    if c in ('+', '-'):
        return True
    return _is_unicode_flag_at(s, pos)


def _can_start_offset(s: str, pos: int) -> bool:
    """Return True if the character at *pos* could begin an offset."""
    if pos >= len(s):
        return False
    return s[pos].isdigit() or s[pos] in ('-', '.')


def _build_line_starts(content: bytes) -> List[int]:
    """Return the byte offset of the start of each line.

    Lines are delimited by ``\\r?\\n``.  The first element is always 0.
    Each subsequent element is the byte offset immediately after a ``\\n``
    (or ``\\r\\n``) sequence.

    Args:
        content: Raw file content as bytes.

    Returns:
        List of byte offsets, one per line start.
    """
    starts = [0]
    for m in re.finditer(b'\r?\n', content):
        starts.append(m.end())
    return starts


def _count_logical_lines(line_starts: List[int], total: int) -> int:
    """Return the number of logical lines for index resolution.

    When content ends with a newline the trailing empty entry at position
    *total* is not counted as a logical line, so that negative indices
    behave consistently with content that lacks a trailing newline.

    Args:
        line_starts: Line-start offsets from :func:`_build_line_starts`.
        total: Total byte length of the content.

    Returns:
        Number of logical lines.
    """
    n = len(line_starts)
    if n > 0 and line_starts[-1] == total:
        return n - 1
    return n


def _resolve_offset(
    offset: Optional[Offset],
    content: bytes,
    line_starts: List[int],
    n_logical: int,
    total: int,
    encoding: str,
    is_start: bool,
) -> int:
    """Resolve an Offset to an absolute byte position.

    Args:
        offset: The Offset to resolve, or None for a file boundary.
        content: Raw file content as bytes.
        line_starts: Line-start offsets from :func:`_build_line_starts`.
        n_logical: Logical line count from :func:`_count_logical_lines`.
        total: Total byte length of *content*.
        encoding: Character encoding for Unicode offset mode.
        is_start: True when resolving the start of a range.

    Returns:
        Absolute byte position (not yet clamped to ``[0, total]``).
    """
    if offset is None:
        return 0 if is_start else total

    if offset.line_no is None:
        line_base = 0
    else:
        line_no = offset.line_no
        if line_no < 0:
            line_no = n_logical + line_no
        line_no = max(0, min(line_no, n_logical))
        line_base = line_starts[line_no] if line_no < len(line_starts) else total

    if offset.char_offset is None:
        return line_base

    char_off = offset.char_offset

    if offset.is_unicode:
        return _unicode_offset_to_byte(content, line_base, char_off, encoding)

    return line_base + char_off


def _unicode_offset_to_byte(
    content: bytes,
    line_base: int,
    char_off: int,
    encoding: str,
) -> int:
    """Convert a Unicode character offset relative to *line_base* to bytes.

    The offset is measured within the current line: the line ends at the
    first ``\\r?\\n`` sequence after *line_base* (or at the end of
    *content* if no newline follows).

    Negative *char_off* values index from the end of the line
    (``max(0, n_chars + char_off)``).

    Note: byte position is computed by re-encoding the decoded prefix.
    For files that are valid in the specified *encoding* this is exact.
    Files with invalid byte sequences may yield approximate positions.

    Args:
        content: Raw file content as bytes.
        line_base: Absolute byte position of the line start.
        char_off: Unicode character offset from line start (negative indexes
            from the line end).
        encoding: Character encoding.

    Returns:
        Absolute byte position.
    """
    line_content = content[line_base:]
    m = re.search(b'\r?\n', line_content)
    if m:
        line_content = line_content[:m.start()]

    try:
        text = line_content.decode(encoding, errors='replace')
    except LookupError:
        return line_base

    n_chars = len(text)
    if char_off < 0:
        char_off = max(0, n_chars + char_off)
    else:
        char_off = min(char_off, n_chars)

    if char_off == 0:
        return line_base
    if char_off >= n_chars:
        return line_base + len(line_content)

    byte_off = len(text[:char_off].encode(encoding, errors='replace'))
    return line_base + byte_off
