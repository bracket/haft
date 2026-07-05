"""Tests for the haft selection module."""

import pytest

from haft.selection import Offset, Range, Selection


# ---------------------------------------------------------------------------
# Offset tests
# ---------------------------------------------------------------------------

class TestOffsetParse:
    """Tests for Offset.parse."""

    def test_line_only(self):
        """Parse a bare line number."""
        offset, pos = Offset.parse("10", 0)
        assert offset.line_no == 10
        assert offset.char_offset is None
        assert offset.is_unicode is False
        assert pos == 2

    def test_line_zero(self):
        """Parse line number zero."""
        offset, pos = Offset.parse("0", 0)
        assert offset.line_no == 0
        assert offset.char_offset is None
        assert pos == 1

    def test_negative_line(self):
        """Parse a negative (Python-style) line number."""
        offset, pos = Offset.parse("-1", 0)
        assert offset.line_no == -1
        assert offset.char_offset is None
        assert pos == 2

    def test_negative_large_line(self):
        """Parse a large negative line number."""
        offset, pos = Offset.parse("-100", 0)
        assert offset.line_no == -100
        assert pos == 4

    def test_line_and_byte_offset(self):
        """Parse line number with a byte character offset."""
        offset, pos = Offset.parse("0.5", 0)
        assert offset.line_no == 0
        assert offset.char_offset == 5
        assert offset.is_unicode is False
        assert pos == 3

    def test_line_and_unicode_offset(self):
        """Parse line number with a Unicode character offset."""
        offset, pos = Offset.parse("0.u5", 0)
        assert offset.line_no == 0
        assert offset.char_offset == 5
        assert offset.is_unicode is True
        assert pos == 4

    def test_line_and_zero_byte_offset(self):
        """Parse line number with a zero byte offset."""
        offset, pos = Offset.parse("1.0", 0)
        assert offset.line_no == 1
        assert offset.char_offset == 0
        assert offset.is_unicode is False
        assert pos == 3

    def test_line_and_zero_unicode_offset(self):
        """Parse line number with a zero Unicode offset."""
        offset, pos = Offset.parse("1.u0", 0)
        assert offset.line_no == 1
        assert offset.char_offset == 0
        assert offset.is_unicode is True
        assert pos == 4

    def test_line_with_dot_no_offset(self):
        """Parse a line number followed by a dot but no char offset."""
        offset, pos = Offset.parse("3.", 0)
        assert offset.line_no == 3
        assert offset.char_offset is None
        assert offset.is_unicode is False
        assert pos == 2

    def test_negative_line_and_byte_offset(self):
        """Parse a negative line number with a byte offset."""
        offset, pos = Offset.parse("-1.10", 0)
        assert offset.line_no == -1
        assert offset.char_offset == 10
        assert offset.is_unicode is False
        assert pos == 5

    def test_file_level_byte_offset(self):
        """Parse a file-level byte offset (no line number)."""
        offset, pos = Offset.parse(".100", 0)
        assert offset.line_no is None
        assert offset.char_offset == 100
        assert offset.is_unicode is False
        assert pos == 4

    def test_file_level_unicode_offset(self):
        """Parse a file-level Unicode character offset."""
        offset, pos = Offset.parse(".u50", 0)
        assert offset.line_no is None
        assert offset.char_offset == 50
        assert offset.is_unicode is True
        assert pos == 4

    def test_parse_with_start_pos(self):
        """Parse an offset starting at a non-zero position."""
        offset, pos = Offset.parse("XX5.3", 2)
        assert offset.line_no == 5
        assert offset.char_offset == 3
        assert pos == 5

    def test_stops_at_colon(self):
        """Parsing stops at a colon separator."""
        offset, pos = Offset.parse("10:20", 0)
        assert offset.line_no == 10
        assert pos == 2  # stopped before ':'

    def test_stops_at_comma(self):
        """Parsing stops at a comma separator."""
        offset, pos = Offset.parse("5,6", 0)
        assert offset.line_no == 5
        assert pos == 1

    def test_empty_string_raises(self):
        """Parsing an empty string raises ValueError."""
        with pytest.raises(ValueError, match="end of string"):
            Offset.parse("", 0)

    def test_invalid_start_raises(self):
        """An invalid starting character raises ValueError."""
        with pytest.raises(ValueError):
            Offset.parse("abc", 0)

    def test_dot_without_digits_raises(self):
        """A dot with no following digits raises ValueError."""
        with pytest.raises(ValueError, match="Expected digits"):
            Offset.parse(".", 0)

    def test_dot_u_without_digits_does_not_consume_u(self):
        """A dot followed by 'u' then non-digit should raise ValueError."""
        with pytest.raises(ValueError):
            Offset.parse(".ux", 0)

    def test_minus_without_digits_raises(self):
        """A minus sign with no following digits raises ValueError."""
        with pytest.raises(ValueError):
            Offset.parse("-", 0)

    def test_negative_byte_char_offset(self):
        """Parse a line+negative byte char offset."""
        offset, pos = Offset.parse("1.-10", 0)
        assert offset.line_no == 1
        assert offset.char_offset == -10
        assert offset.is_unicode is False
        assert pos == 5

    def test_negative_unicode_char_offset(self):
        """Parse a line+negative Unicode char offset."""
        offset, pos = Offset.parse("1.-u10", 0)
        assert offset.line_no == 1
        assert offset.char_offset == -10
        assert offset.is_unicode is True
        assert pos == 6

    def test_explicit_positive_byte_char_offset(self):
        """Explicit '+' sign is accepted and equivalent to no sign."""
        offset, pos = Offset.parse("1.+10", 0)
        assert offset.line_no == 1
        assert offset.char_offset == 10
        assert offset.is_unicode is False
        assert pos == 5

    def test_explicit_positive_unicode_char_offset(self):
        """Explicit '+' with unicode flag is accepted."""
        offset, pos = Offset.parse("1.+u10", 0)
        assert offset.line_no == 1
        assert offset.char_offset == 10
        assert offset.is_unicode is True
        assert pos == 6

    def test_file_level_negative_byte_offset(self):
        """Parse a file-level negative byte offset."""
        offset, pos = Offset.parse(".-100", 0)
        assert offset.line_no is None
        assert offset.char_offset == -100
        assert offset.is_unicode is False
        assert pos == 5

    def test_file_level_negative_unicode_offset(self):
        """Parse a file-level negative Unicode offset."""
        offset, pos = Offset.parse(".-u50", 0)
        assert offset.line_no is None
        assert offset.char_offset == -50
        assert offset.is_unicode is True
        assert pos == 5

    def test_negative_zero_char_offset(self):
        """Parse a negative-zero char offset (normalises to 0)."""
        offset, pos = Offset.parse("0.-0", 0)
        assert offset.char_offset == 0

    def test_sign_without_digits_raises(self):
        """A sign with no following digits raises ValueError."""
        with pytest.raises(ValueError, match="Expected digit or 'u' flag after sign"):
            Offset.parse("0.+", 0)

    def test_sign_with_only_u_raises(self):
        """A sign followed by 'u' but no digit raises ValueError."""
        with pytest.raises(ValueError, match="Expected digit after 'u' flag"):
            Offset.parse("0.-u", 0)


class TestOffsetStr:
    """Tests for Offset.__str__."""

    def test_line_only(self):
        assert str(Offset(line_no=0)) == "0"

    def test_negative_line(self):
        assert str(Offset(line_no=-1)) == "-1"

    def test_line_and_byte_offset(self):
        assert str(Offset(line_no=0, char_offset=5)) == "0.5"

    def test_line_and_unicode_offset(self):
        assert str(Offset(line_no=0, char_offset=5, is_unicode=True)) == "0.u5"

    def test_file_level_byte_offset(self):
        assert str(Offset(char_offset=100)) == ".100"

    def test_file_level_unicode_offset(self):
        assert str(Offset(char_offset=50, is_unicode=True)) == ".u50"

    def test_negative_byte_char_offset(self):
        assert str(Offset(line_no=1, char_offset=-10)) == "1.-10"

    def test_negative_unicode_char_offset(self):
        assert str(Offset(line_no=1, char_offset=-10, is_unicode=True)) == "1.-u10"

    def test_positive_char_offset_no_sign(self):
        assert str(Offset(line_no=0, char_offset=5)) == "0.5"

    def test_file_level_negative_offset(self):
        assert str(Offset(char_offset=-100)) == ".-100"


class TestOffsetRoundTrip:
    """Round-trip tests for Offset (parse → str → parse)."""

    def _roundtrip(self, s: str) -> str:
        offset, _ = Offset.parse(s, 0)
        return str(offset)

    def test_line_only(self):
        assert self._roundtrip("10") == "10"

    def test_negative_line(self):
        assert self._roundtrip("-1") == "-1"

    def test_line_and_byte(self):
        assert self._roundtrip("0.5") == "0.5"

    def test_line_and_unicode(self):
        assert self._roundtrip("0.u5") == "0.u5"

    def test_file_level_byte(self):
        assert self._roundtrip(".100") == ".100"

    def test_file_level_unicode(self):
        assert self._roundtrip(".u50") == ".u50"

    def test_negative_byte_char_offset(self):
        assert self._roundtrip("1.-10") == "1.-10"

    def test_negative_unicode_char_offset(self):
        assert self._roundtrip("1.-u10") == "1.-u10"

    def test_explicit_plus_normalised(self):
        assert self._roundtrip("1.+10") == "1.10"

    def test_explicit_plus_unicode_normalised(self):
        assert self._roundtrip("1.+u10") == "1.u10"

    def test_file_level_negative(self):
        assert self._roundtrip(".-100") == ".-100"


class TestOffsetInit:
    """Tests for Offset constructor validation."""

    def test_both_none_raises(self):
        with pytest.raises(ValueError, match="at least a line number"):
            Offset()

    def test_line_no_only(self):
        o = Offset(line_no=5)
        assert o.line_no == 5
        assert o.char_offset is None
        assert o.is_unicode is False

    def test_char_offset_only(self):
        o = Offset(char_offset=10)
        assert o.line_no is None
        assert o.char_offset == 10


class TestOffsetEquality:
    """Tests for Offset equality."""

    def test_equal(self):
        assert Offset(line_no=0) == Offset(line_no=0)

    def test_not_equal_line(self):
        assert Offset(line_no=0) != Offset(line_no=1)

    def test_not_equal_char_offset(self):
        assert Offset(line_no=0, char_offset=5) != Offset(line_no=0, char_offset=6)

    def test_not_equal_unicode(self):
        a = Offset(line_no=0, char_offset=5, is_unicode=False)
        b = Offset(line_no=0, char_offset=5, is_unicode=True)
        assert a != b

    def test_not_equal_type(self):
        assert Offset(line_no=0).__eq__("0") is NotImplemented


# ---------------------------------------------------------------------------
# Range tests
# ---------------------------------------------------------------------------

class TestRangeParse:
    """Tests for Range.parse."""

    def test_full_range_lines(self):
        r, pos = Range.parse("0:10", 0)
        assert r.start == Offset(line_no=0)
        assert r.end == Offset(line_no=10)
        assert pos == 4

    def test_full_range_with_byte_offsets(self):
        r, pos = Range.parse("0.5:1.0", 0)
        assert r.start == Offset(line_no=0, char_offset=5)
        assert r.end == Offset(line_no=1, char_offset=0)
        assert pos == 7

    def test_full_range_with_unicode_offsets(self):
        r, pos = Range.parse("0.u5:1.u0", 0)
        assert r.start == Offset(line_no=0, char_offset=5, is_unicode=True)
        assert r.end == Offset(line_no=1, char_offset=0, is_unicode=True)
        assert pos == 9

    def test_start_only(self):
        r, pos = Range.parse("10:", 0)
        assert r.start == Offset(line_no=10)
        assert r.end is None
        assert pos == 3

    def test_end_only(self):
        r, pos = Range.parse(":10", 0)
        assert r.start is None
        assert r.end == Offset(line_no=10)
        assert pos == 3

    def test_whole_file(self):
        r, pos = Range.parse(":", 0)
        assert r.start is None
        assert r.end is None
        assert pos == 1

    def test_stops_at_comma(self):
        r, pos = Range.parse("0:10,20:30", 0)
        assert r.start == Offset(line_no=0)
        assert r.end == Offset(line_no=10)
        assert pos == 4

    def test_negative_line_in_range(self):
        r, pos = Range.parse("-1:", 0)
        assert r.start == Offset(line_no=-1)
        assert r.end is None

    def test_missing_colon_raises(self):
        with pytest.raises(ValueError, match="Expected ':'"):
            Range.parse("10", 0)

    def test_invalid_end_character_raises(self):
        with pytest.raises(ValueError):
            Range.parse(":!", 0)

    def test_zero_length_range(self):
        r, pos = Range.parse("5:5", 0)
        assert r.start == Offset(line_no=5)
        assert r.end == Offset(line_no=5)


class TestRangeStr:
    """Tests for Range.__str__."""

    def test_full_range(self):
        assert str(Range(Offset(line_no=0), Offset(line_no=10))) == "0:10"

    def test_start_only(self):
        assert str(Range(Offset(line_no=10), None)) == "10:"

    def test_end_only(self):
        assert str(Range(None, Offset(line_no=10))) == ":10"

    def test_whole_file(self):
        assert str(Range(None, None)) == ":"

    def test_with_char_offsets(self):
        start = Offset(line_no=0, char_offset=5)
        end = Offset(line_no=1, char_offset=0)
        assert str(Range(start, end)) == "0.5:1.0"

    def test_with_unicode_offsets(self):
        start = Offset(line_no=0, char_offset=5, is_unicode=True)
        end = Offset(line_no=1, char_offset=0, is_unicode=True)
        assert str(Range(start, end)) == "0.u5:1.u0"


class TestRangeEquality:
    """Tests for Range equality."""

    def test_equal(self):
        r1 = Range(Offset(line_no=0), Offset(line_no=10))
        r2 = Range(Offset(line_no=0), Offset(line_no=10))
        assert r1 == r2

    def test_not_equal_start(self):
        r1 = Range(Offset(line_no=0), Offset(line_no=10))
        r2 = Range(Offset(line_no=1), Offset(line_no=10))
        assert r1 != r2

    def test_not_equal_end(self):
        r1 = Range(Offset(line_no=0), Offset(line_no=10))
        r2 = Range(Offset(line_no=0), Offset(line_no=11))
        assert r1 != r2

    def test_not_equal_type(self):
        r = Range(Offset(line_no=0), None)
        assert r.__eq__("0:") is NotImplemented


# ---------------------------------------------------------------------------
# Selection tests
# ---------------------------------------------------------------------------

class TestSelectionParse:
    """Tests for Selection.parse."""

    def test_single_range(self):
        sel = Selection.parse("0:10")
        assert len(sel.ranges) == 1
        assert sel.ranges[0].start == Offset(line_no=0)
        assert sel.ranges[0].end == Offset(line_no=10)

    def test_multiple_ranges(self):
        sel = Selection.parse("0:10,20:30")
        assert len(sel.ranges) == 2
        assert sel.ranges[0] == Range(Offset(line_no=0), Offset(line_no=10))
        assert sel.ranges[1] == Range(Offset(line_no=20), Offset(line_no=30))

    def test_three_ranges(self):
        sel = Selection.parse("0:5,10:15,20:25")
        assert len(sel.ranges) == 3

    def test_start_only(self):
        sel = Selection.parse("10:")
        assert sel.ranges[0].start == Offset(line_no=10)
        assert sel.ranges[0].end is None

    def test_end_only(self):
        sel = Selection.parse(":10")
        assert sel.ranges[0].start is None
        assert sel.ranges[0].end == Offset(line_no=10)

    def test_byte_offsets(self):
        sel = Selection.parse("0.5:1.0")
        assert sel.ranges[0].start == Offset(line_no=0, char_offset=5)
        assert sel.ranges[0].end == Offset(line_no=1, char_offset=0)

    def test_unicode_offsets(self):
        sel = Selection.parse("0.u5:1.u0")
        assert sel.ranges[0].start == Offset(line_no=0, char_offset=5, is_unicode=True)
        assert sel.ranges[0].end == Offset(line_no=1, char_offset=0, is_unicode=True)

    def test_negative_line_number(self):
        sel = Selection.parse("-1:")
        assert sel.ranges[0].start == Offset(line_no=-1)
        assert sel.ranges[0].end is None

    def test_file_level_offsets(self):
        sel = Selection.parse(".0:.100")
        assert sel.ranges[0].start == Offset(char_offset=0)
        assert sel.ranges[0].end == Offset(char_offset=100)

    def test_mixed_ranges(self):
        sel = Selection.parse(":10,20:")
        assert sel.ranges[0].start is None
        assert sel.ranges[0].end == Offset(line_no=10)
        assert sel.ranges[1].start == Offset(line_no=20)
        assert sel.ranges[1].end is None

    def test_whitespace_stripped(self):
        sel = Selection.parse("  0:10  ")
        assert len(sel.ranges) == 1

    def test_space_after_comma(self):
        sel = Selection.parse("0:10, 20:30")
        assert len(sel.ranges) == 2
        assert sel.ranges[1].start == Offset(line_no=20)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Selection.parse("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Selection.parse("   ")

    def test_invalid_syntax_raises(self):
        with pytest.raises(ValueError):
            Selection.parse("abc")

    def test_trailing_comma_raises(self):
        with pytest.raises(ValueError):
            Selection.parse("0:10,")


class TestSelectionStr:
    """Tests for Selection.__str__."""

    def test_single_range(self):
        sel = Selection([Range(Offset(line_no=0), Offset(line_no=10))])
        assert str(sel) == "0:10"

    def test_multiple_ranges(self):
        sel = Selection([
            Range(Offset(line_no=0), Offset(line_no=10)),
            Range(Offset(line_no=20), Offset(line_no=30)),
        ])
        assert str(sel) == "0:10,20:30"

    def test_start_only(self):
        sel = Selection([Range(Offset(line_no=10), None)])
        assert str(sel) == "10:"

    def test_end_only(self):
        sel = Selection([Range(None, Offset(line_no=10))])
        assert str(sel) == ":10"


class TestSelectionRoundTrip:
    """Round-trip tests for Selection (parse → str → parse)."""

    def _roundtrip(self, s: str) -> str:
        return str(Selection.parse(s))

    def test_single_line_range(self):
        assert self._roundtrip("0:10") == "0:10"

    def test_multi_range(self):
        assert self._roundtrip("0:10,20:30") == "0:10,20:30"

    def test_start_only(self):
        assert self._roundtrip("10:") == "10:"

    def test_end_only(self):
        assert self._roundtrip(":10") == ":10"

    def test_byte_offsets(self):
        assert self._roundtrip("0.5:1.0") == "0.5:1.0"

    def test_unicode_offsets(self):
        assert self._roundtrip("0.u5:1.u0") == "0.u5:1.u0"

    def test_negative_line(self):
        assert self._roundtrip("-1:") == "-1:"

    def test_whole_file(self):
        assert self._roundtrip(":") == ":"

    def test_file_level_offsets(self):
        assert self._roundtrip(".0:.100") == ".0:.100"

    def test_three_ranges(self):
        assert self._roundtrip("0:5,10:15,20:25") == "0:5,10:15,20:25"


class TestSelectionEquality:
    """Tests for Selection equality."""

    def test_equal(self):
        s1 = Selection.parse("0:10")
        s2 = Selection.parse("0:10")
        assert s1 == s2

    def test_not_equal_ranges(self):
        s1 = Selection.parse("0:10")
        s2 = Selection.parse("0:20")
        assert s1 != s2

    def test_not_equal_type(self):
        s = Selection.parse("0:10")
        assert s.__eq__("0:10") is NotImplemented


class TestRepresentations:
    """Tests for __repr__ methods."""

    def test_offset_repr(self):
        o = Offset(line_no=0, char_offset=5, is_unicode=True)
        assert repr(o) == "Offset(line_no=0, char_offset=5, is_unicode=True)"

    def test_range_repr(self):
        r = Range(Offset(line_no=0), Offset(line_no=10))
        assert "Range(" in repr(r)
        assert "start=" in repr(r)
        assert "end=" in repr(r)

    def test_selection_repr(self):
        s = Selection.parse("0:10")
        assert "Selection(" in repr(s)
        assert "ranges=" in repr(s)


# ---------------------------------------------------------------------------
# Resolution / extraction tests (public API)
# ---------------------------------------------------------------------------

def _enc(s: str, encoding: str = 'utf-8') -> bytes:
    return s.encode(encoding)


class TestRangeResolve:
    """Tests for Range.resolve."""

    def test_line_range(self):
        """Line range resolves to correct byte offsets."""
        content = _enc("line0\nline1\nline2\n")
        r = Range(Offset(line_no=1), Offset(line_no=2))
        start, end = r.resolve(content)
        assert content[start:end] == b"line1\n"

    def test_whole_file(self):
        content = _enc("hello\nworld\n")
        r = Range(None, None)
        start, end = r.resolve(content)
        assert start == 0
        assert end == len(content)

    def test_byte_offset_within_line(self):
        content = _enc("hello world\n")
        r = Range(Offset(line_no=0, char_offset=6), Offset(line_no=0, char_offset=11))
        start, end = r.resolve(content)
        assert content[start:end] == b"world"

    def test_cross_line_byte_range(self):
        content = _enc("hello\nworld\n")
        r = Range(Offset(line_no=0, char_offset=3), Offset(line_no=1, char_offset=3))
        start, end = r.resolve(content)
        assert content[start:end] == b"lo\nwor"

    def test_negative_line_no(self):
        content = _enc("a\nb\nc\nd\n")
        r = Range(Offset(line_no=-2), Offset(line_no=-1))
        start, end = r.resolve(content)
        assert content[start:end] == b"c\n"

    def test_clamped_out_of_bounds(self):
        content = _enc("only line\n")
        r = Range(Offset(line_no=0), Offset(line_no=100))
        start, end = r.resolve(content)
        assert start == 0
        assert end == len(content)

    def test_negative_length_produces_empty(self):
        content = _enc("hello\n")
        r = Range(Offset(line_no=1), Offset(line_no=0))
        start, end = r.resolve(content)
        assert start == end

    def test_file_level_byte_offset(self):
        content = _enc("hello world")
        r = Range(Offset(char_offset=6), Offset(char_offset=11))
        start, end = r.resolve(content)
        assert content[start:end] == b"world"


class TestRangeExtract:
    """Tests for Range.extract: result equals content[start:end].decode(...)."""

    def test_same_line_byte_range(self):
        content = _enc("hello world\n")
        r = Range(Offset(line_no=0, char_offset=6), Offset(line_no=0, char_offset=11))
        assert r.extract(content) == "world"

    def test_cross_line_range(self):
        content = _enc("hello\nworld\n")
        r = Range(Offset(line_no=0, char_offset=3), Offset(line_no=1, char_offset=3))
        result = r.extract(content)
        start, end = r.resolve(content)
        assert result == content[start:end].decode('utf-8', errors='replace')

    def test_line_range(self):
        content = _enc("line0\nline1\nline2\n")
        r = Range(Offset(line_no=1), Offset(line_no=3))
        result = r.extract(content)
        start, end = r.resolve(content)
        assert result == content[start:end].decode('utf-8', errors='replace')

    def test_empty_range(self):
        content = _enc("hello\n")
        r = Range(Offset(line_no=0, char_offset=3), Offset(line_no=0, char_offset=3))
        assert r.extract(content) == ""

    def test_negative_length_range(self):
        content = _enc("hello\n")
        r = Range(Offset(line_no=0, char_offset=5), Offset(line_no=0, char_offset=2))
        assert r.extract(content) == ""


class TestSelectionResolve:
    """Tests for Selection.resolve."""

    def test_single_range(self):
        content = _enc("a\nb\nc\n")
        sel = Selection.parse("1:2")
        pairs = sel.resolve(content)
        assert len(pairs) == 1
        assert content[pairs[0][0]:pairs[0][1]] == b"b\n"

    def test_multi_range(self):
        content = _enc("a\nb\nc\nd\n")
        sel = Selection.parse("0:1,2:3")
        pairs = sel.resolve(content)
        assert len(pairs) == 2
        assert content[pairs[0][0]:pairs[0][1]] == b"a\n"
        assert content[pairs[1][0]:pairs[1][1]] == b"c\n"

    def test_line_starts_reused(self):
        """Multiple ranges in a selection reuse the same line index."""
        content = _enc("line0\nline1\nline2\nline3\n")
        sel = Selection.parse("0:2,2:4")
        pairs = sel.resolve(content)
        assert len(pairs) == 2
        assert content[pairs[0][0]:pairs[0][1]] == b"line0\nline1\n"
        assert content[pairs[1][0]:pairs[1][1]] == b"line2\nline3\n"


class TestSelectionExtract:
    """Tests for Selection.extract."""

    def test_single_range_equality(self):
        """Extract must equal content[start:end].decode(...)."""
        content = _enc("hello world\nfoo bar\n")
        sel = Selection.parse("0.6:0.11")
        result = sel.extract(content)
        pairs = sel.resolve(content)
        expected = content[pairs[0][0]:pairs[0][1]].decode('utf-8', errors='replace')
        assert result == expected

    def test_multi_range_joiner(self):
        """Multiple ranges are joined by the joiner."""
        content = _enc("a\nb\nc\n")
        sel = Selection.parse("0:1,2:3")
        result = sel.extract(content, joiner='---')
        assert result == "a\n---c\n"

    def test_empty_joiner(self):
        """Empty joiner concatenates ranges directly."""
        content = _enc("aaa\nbbb\nccc\n")
        sel = Selection.parse("0:1,2:3")
        result = sel.extract(content, joiner='')
        assert result == "aaa\nccc\n"

    def test_negative_length_range_empty(self):
        content = _enc("hello\n")
        sel = Selection.parse("1:0")
        assert sel.extract(content) == ""


# ---------------------------------------------------------------------------
# Bug-fix regression tests
# ---------------------------------------------------------------------------

class TestTrailingNewline:
    """Regression tests: content with trailing newline."""

    def test_last_line_includes_newline(self):
        content = _enc("line0\nline1\nline2\n")
        r = Range(Offset(line_no=1), Offset(line_no=2))
        result = r.extract(content)
        start, end = r.resolve(content)
        assert result == content[start:end].decode('utf-8', errors='replace')
        assert result == "line1\n"

    def test_negative_one_is_last_real_line(self):
        """With trailing newline, -1 resolves to the last real line."""
        content = _enc("a\nb\nc\n")
        r = Range(Offset(line_no=-1), None)
        start, end = r.resolve(content)
        assert content[start:end] == b"c\n"

    def test_all_lines_with_trailing_newline(self):
        content = _enc("x\ny\nz\n")
        r = Range(None, None)
        start, end = r.resolve(content)
        assert start == 0
        assert end == len(content)


class TestCRLFLineEndings:
    """Regression tests: content with \\r\\n line endings."""

    def test_line_range_crlf(self):
        """Line range on CRLF content does not add spurious \\n."""
        content = b"line0\r\nline1\r\nline2\r\n"
        r = Range(Offset(line_no=1), Offset(line_no=2))
        result = r.extract(content)
        start, end = r.resolve(content)
        assert result == content[start:end].decode('utf-8', errors='replace')
        assert result == "line1\r\n"

    def test_byte_offset_crlf(self):
        """Byte offset into CRLF content gives true byte position."""
        content = b"abc\r\ndef\r\n"
        r = Range(Offset(line_no=0, char_offset=1), Offset(line_no=0, char_offset=3))
        result = r.extract(content)
        start, end = r.resolve(content)
        assert result == content[start:end].decode('utf-8', errors='replace')
        assert result == "bc"

    def test_cross_line_crlf(self):
        """Cross-line range on CRLF content is correct."""
        content = b"hello\r\nworld\r\n"
        r = Range(Offset(line_no=0, char_offset=3), Offset(line_no=1, char_offset=3))
        result = r.extract(content)
        start, end = r.resolve(content)
        assert result == content[start:end].decode('utf-8', errors='replace')
        assert result == "lo\r\nwor"


class TestMultibyteUTF8:
    """Regression tests: multibyte UTF-8 content."""

    def test_unicode_offset_multibyte(self):
        """Unicode offset counts characters, not bytes, in multibyte content."""
        # 'é' is 2 bytes in UTF-8
        content = "héllo".encode('utf-8')
        r = Range(Offset(line_no=0, char_offset=0, is_unicode=True),
                  Offset(line_no=0, char_offset=3, is_unicode=True))
        result = r.extract(content)
        start, end = r.resolve(content)
        assert result == content[start:end].decode('utf-8', errors='replace')
        assert result == "hél"

    def test_byte_offset_multibyte(self):
        """Byte offset into multibyte content gives raw byte slice."""
        content = "héllo".encode('utf-8')
        # 'h' = 1 byte, 'é' = 2 bytes (0xC3 0xA9), so byte 1 starts mid-char
        r = Range(Offset(line_no=0, char_offset=3), Offset(line_no=0, char_offset=5))
        result = r.extract(content)
        start, end = r.resolve(content)
        assert result == content[start:end].decode('utf-8', errors='replace')

    def test_extract_equals_slice_decode(self):
        """For any single range, extract equals content[start:end].decode(...)."""
        content = "日本語テスト\nfoo\n".encode('utf-8')
        r = Range(Offset(line_no=0), Offset(line_no=1))
        result = r.extract(content)
        start, end = r.resolve(content)
        assert result == content[start:end].decode('utf-8', errors='replace')

    def test_unicode_offset_latin1(self):
        """Unicode offset works with latin-1 encoding."""
        content = "caf\xe9\n".encode('latin-1')
        # 'caf\xe9' is 4 chars, each 1 byte in latin-1
        r = Range(Offset(line_no=0, char_offset=0, is_unicode=True),
                  Offset(line_no=0, char_offset=3, is_unicode=True))
        result = r.extract(content, encoding='latin-1')
        start, end = r.resolve(content, encoding='latin-1')
        assert result == content[start:end].decode('latin-1', errors='replace')
        assert result == "caf"


class TestEdgeCases:
    """Edge cases for resolution."""

    def test_empty_content(self):
        content = b""
        r = Range(None, None)
        start, end = r.resolve(content)
        assert start == 0
        assert end == 0

    def test_single_char_no_newline(self):
        content = b"x"
        r = Range(Offset(line_no=0), Offset(line_no=1))
        start, end = r.resolve(content)
        assert start == 0
        assert end == 1

    def test_line_starts_at_correct_byte_after_crlf(self):
        """Line 1 starts right after \\r\\n."""
        content = b"foo\r\nbar"
        r = Range(Offset(line_no=1), None)
        start, end = r.resolve(content)
        assert content[start:end] == b"bar"

    def test_file_level_byte_offset_range(self):
        content = b"abcdefgh"
        r = Range(Offset(char_offset=2), Offset(char_offset=5))
        start, end = r.resolve(content)
        assert content[start:end] == b"cde"
