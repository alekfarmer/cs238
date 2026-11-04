"""Tests for reading and writing .gph files."""

from pathlib import Path

import pytest

from project1 import read_gph, to_digraph, write_gph

NAMES = ("a", "b", "c")


def test_round_trip(tmp_path: Path) -> None:
    parents = [frozenset(), frozenset({0}), frozenset({0, 1})]
    path = tmp_path / "g.gph"
    write_gph(to_digraph(parents), NAMES, path)
    assert read_gph(path, NAMES) == parents
    assert path.read_text() == "a, b\na, c\nb, c\n"


def test_reads_course_format_without_spaces_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "g.gph"
    path.write_text("a,c\n\nb,c\n")
    assert read_gph(path, NAMES) == [frozenset(), frozenset(), frozenset({0, 1})]


@pytest.mark.parametrize(("line", "message"), [("a,z\n", "unknown"), ("a\n", "expected")])
def test_rejects_bad_lines(tmp_path: Path, line: str, message: str) -> None:
    path = tmp_path / "g.gph"
    path.write_text(line)
    with pytest.raises(ValueError, match=message):
        read_gph(path, NAMES)
