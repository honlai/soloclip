"""Selection is where the brief is actually enforced - cover the decisions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soloclip.select import _assemble, _score, _snap_span, _trim_end_to  # noqa: E402


class FakeConfig:
    def __init__(self, **over):
        self.values = {
            "select.join_penalty_seconds": 1.5,
            "select.gap_penalty_per_min": 0.5,
            "select.target_seconds": 20.0,
        }
        self.values.update(over)

    def get(self, key, default=None):
        return self.values.get(key, default)


def words(pairs):
    return [{"start": s, "end": e, "word": "w"} for s, e in pairs]


def sentences(pairs):
    return [{"start": s, "end": e, "text": "s."} for s, e in pairs]


def test_trim_picks_boundary_nearest_the_target():
    w = words([(0, 1), (1, 2), (2, 3), (3, 4)])
    # target 2.5s, no headroom -> the 2.0 boundary is the nearest one allowed
    assert _trim_end_to((0.0, 4.0), 2.5, 2.5, w, []) == (0.0, 2.0)


def test_trim_is_noop_when_short_enough():
    assert _trim_end_to((0.0, 3.0), 5.0, 6.0, words([(0, 3)]), []) == (0.0, 3.0)


def test_trim_overshoots_target_to_finish_the_sentence():
    """20s target, 24s ceiling: a sentence ending at 22s beats cutting at 20s."""
    w = words([(t, t + 1.0) for t in range(0, 30)])
    sent = sentences([(0.0, 12.0), (12.0, 22.0), (22.0, 28.0)])
    assert _trim_end_to((0.0, 28.0), 20.0, 24.0, w, sent) == (0.0, 22.0)


def test_trim_never_passes_the_hard_ceiling():
    w = words([(t, t + 1.0) for t in range(0, 40)])
    sent = sentences([(0.0, 26.0)])  # only boundary is past the ceiling
    _, end = _trim_end_to((0.0, 40.0), 20.0, 24.0, w, sent)
    assert end <= 24.0


def test_assemble_stops_at_target():
    pieces = [(0.0, 12.0), (30.0, 40.0)]
    w = words([(t, t + 1) for t in range(0, 45)])
    chosen, used = _assemble(pieces, 20.0, 24.0, 3.0, w, [])
    assert 20.0 - 1e-6 <= used <= 24.0 + 1e-6
    assert len(chosen) == 2


def test_assemble_drops_piece_that_would_be_a_fragment():
    """A 1s tail is worse than 19s clean - min_piece must reject it."""
    pieces = [(0.0, 19.0), (30.0, 32.0)]
    w = words([(t, t + 1) for t in range(0, 45)])
    chosen, used = _assemble(pieces, 20.0, 24.0, 3.0, w, [])
    assert chosen == [(0.0, 19.0)]
    assert used == 19.0


def test_score_prefers_fewer_joins():
    cfg = FakeConfig()
    single = _score([(0.0, 18.0)], 18.0, cfg)
    spliced = _score([(0.0, 9.0), (60.0, 69.0)], 18.0, cfg)
    assert single > spliced


def test_snap_prefers_sentence_boundaries():
    sentences = [{"start": 2.0, "end": 8.0, "text": "hi."}]
    w = words([(1.0, 2.0), (2.0, 8.0), (8.0, 9.5)])
    got = _snap_span((1.5, 9.0), w, sentences, True, 0.1, 0.25, [(0.0, 20.0)])
    assert got == (1.9, 8.25)


def test_snap_padding_cannot_leave_clean_region():
    sentences = [{"start": 2.0, "end": 8.0, "text": "hi."}]
    w = words([(2.0, 8.0)])
    # clean region ends exactly at 8.0, so the tail pad must not be applied
    got = _snap_span((2.0, 8.0), w, sentences, True, 0.1, 0.25, [(1.0, 8.0)])
    assert got[1] <= 8.0


def test_score_discounts_seconds_past_the_target():
    """Headroom exists to finish a thought, not to pad every clip to the cap."""
    cfg = FakeConfig()
    at_target = _score([(0.0, 20.0)], 20.0, cfg)
    over = _score([(0.0, 24.0)], 24.0, cfg)
    assert over > at_target                 # longer still wins, all else equal
    assert over - at_target < 4.0 * 0.5     # but only weakly


def test_score_still_prefers_target_over_a_longer_spliced_clip():
    cfg = FakeConfig()
    single = _score([(0.0, 20.0)], 20.0, cfg)
    spliced = _score([(0.0, 12.0), (100.0, 112.0)], 24.0, cfg)
    assert single > spliced
