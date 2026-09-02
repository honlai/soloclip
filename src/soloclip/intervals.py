"""Interval-set arithmetic on [start, end] float ranges (seconds).

All functions take/return sorted, non-overlapping lists of (start, end) tuples.
"""

from __future__ import annotations

Interval = tuple[float, float]


def normalize(spans: list[Interval], eps: float = 1e-6) -> list[Interval]:
    """Sort, drop empty spans, and merge anything that touches or overlaps."""
    cleaned = [(float(s), float(e)) for s, e in spans if e - s > eps]
    if not cleaned:
        return []
    cleaned.sort()
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + eps:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def total(spans: list[Interval]) -> float:
    return sum(e - s for s, e in spans)


def intersect(a: list[Interval], b: list[Interval]) -> list[Interval]:
    a, b = normalize(a), normalize(b)
    out: list[Interval] = []
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            out.append((start, end))
        # advance whichever span ends first
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return normalize(out)


def subtract(a: list[Interval], b: list[Interval]) -> list[Interval]:
    a, b = normalize(a), normalize(b)
    out: list[Interval] = []
    for start, end in a:
        cursor = start
        for bs, be in b:
            if be <= cursor or bs >= end:
                continue
            if bs > cursor:
                out.append((cursor, min(bs, end)))
            cursor = max(cursor, be)
            if cursor >= end:
                break
        if cursor < end:
            out.append((cursor, end))
    return normalize(out)


def dilate(spans: list[Interval], pad: float, bounds: Interval | None = None) -> list[Interval]:
    """Grow every span by `pad` on both sides, then re-merge."""
    grown = [(s - pad, e + pad) for s, e in spans]
    if bounds is not None:
        lo, hi = bounds
        grown = [(max(lo, s), min(hi, e)) for s, e in grown]
    return normalize(grown)


def close_gaps(spans: list[Interval], max_gap: float) -> list[Interval]:
    """Bridge gaps no longer than `max_gap`, keeping the silence in between.

    A speaker drawing breath is not a break in the take, and treating it as one
    forces a splice where the original was continuous.
    """
    spans = normalize(spans)
    if max_gap <= 0 or not spans:
        return spans
    out = [spans[0]]
    for start, end in spans[1:]:
        if start - out[-1][1] <= max_gap:
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


def clip(spans: list[Interval], lo: float, hi: float) -> list[Interval]:
    return normalize([(max(lo, s), min(hi, e)) for s, e in spans])


def drop_shorter_than(spans: list[Interval], min_len: float) -> list[Interval]:
    return [(s, e) for s, e in normalize(spans) if e - s >= min_len]


def runs_from_flags(times: list[float], flags: list[bool], step: float) -> list[Interval]:
    """Turn a per-sample boolean mask into intervals covering the true runs."""
    out: list[Interval] = []
    start: float | None = None
    for t, ok in zip(times, flags):
        if ok and start is None:
            start = t
        elif not ok and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, times[-1] + step))
    return normalize(out)
