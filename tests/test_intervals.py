"""Interval arithmetic is the backbone of selection; keep it honest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soloclip.intervals import (  # noqa: E402
    dilate, drop_shorter_than, intersect, normalize, runs_from_flags, subtract, total,
)


def test_normalize_merges_touching():
    assert normalize([(3, 4), (0, 2), (2, 3)]) == [(0.0, 4.0)]


def test_normalize_drops_empty():
    assert normalize([(1, 1), (2, 5)]) == [(2.0, 5.0)]


def test_intersect():
    a = [(0, 10), (20, 30)]
    b = [(5, 25)]
    assert intersect(a, b) == [(5.0, 10.0), (20.0, 25.0)]


def test_subtract_punches_hole():
    assert subtract([(0, 10)], [(3, 5)]) == [(0.0, 3.0), (5.0, 10.0)]


def test_subtract_full_cover():
    assert subtract([(2, 4)], [(0, 10)]) == []


def test_dilate_respects_bounds():
    assert dilate([(1, 2)], 0.5, bounds=(0, 10)) == [(0.5, 2.5)]
    assert dilate([(0.1, 2)], 0.5, bounds=(0, 10)) == [(0.0, 2.5)]


def test_total_and_drop():
    spans = [(0, 3), (10, 20)]
    assert total(spans) == 13.0
    assert drop_shorter_than(spans, 5) == [(10.0, 20.0)]


def test_runs_from_flags():
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    flags = [False, True, True, False, True]
    assert runs_from_flags(times, flags, 0.5) == [(0.5, 1.5), (2.0, 2.5)]


def test_overlap_removal_is_padded():
    """A second voice must be excluded together with its safety pad."""
    dominant = [(0, 60)]
    others = [(30, 31)]
    clean = subtract(dominant, dilate(others, 0.2, bounds=(0, 60)))
    assert clean == [(0.0, 29.8), (31.2, 60.0)]


def test_close_gaps_bridges_breathing_pauses():
    from soloclip.intervals import close_gaps
    spans = [(0, 5), (5.4, 9), (20, 25)]
    assert close_gaps(spans, 1.0) == [(0.0, 9.0), (20.0, 25.0)]


def test_close_gaps_noop_when_disabled():
    from soloclip.intervals import close_gaps
    spans = [(0, 5), (5.4, 9)]
    assert close_gaps(spans, 0.0) == [(0.0, 5.0), (5.4, 9.0)]
