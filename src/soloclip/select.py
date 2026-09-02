"""Stage 6: pick the pieces that become the final clip.

Priority order, per the brief:
  1. one continuous clean stretch, trimmed to target length  (0 joins)
  2. the fewest joins that still get close to the target      (<= max_joins)
  3. nothing at all, if we cannot reach min_seconds
Producing a short or contaminated clip is worse than producing none.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from .config import Config
from .intervals import Interval, intersect, normalize
from .utils import LOG, read_stage, require_stage, write_stage

STAGE = "select"


# -- boundary snapping -------------------------------------------------------

def _snap_span(span: Interval, words: list[dict], sentences: list[dict],
               use_sentences: bool, head_pad: float, tail_pad: float,
               bounds: list[Interval]) -> Interval | None:
    """Pull a span in to the nearest speech boundaries, then pad it back out."""
    start, end = span
    snapped = None
    if use_sentences and sentences:
        inside = [s for s in sentences if s["start"] >= start - 1e-3 and s["end"] <= end + 1e-3]
        if inside:
            snapped = (inside[0]["start"], inside[-1]["end"])
    if snapped is None:
        inside_w = [w for w in words if w["start"] >= start - 1e-3 and w["end"] <= end + 1e-3]
        if not inside_w:
            return None
        snapped = (inside_w[0]["start"], inside_w[-1]["end"])

    padded = (snapped[0] - head_pad, snapped[1] + tail_pad)
    # padding may only reach into audio-clean territory
    allowed = intersect([padded], bounds)
    if not allowed:
        return None
    covering = [a for a in allowed if a[0] <= snapped[0] + 1e-3 and a[1] >= snapped[1] - 1e-3]
    return covering[0] if covering else snapped


def _trim_end_to(span: Interval, target_len: float, max_len: float,
                 words: list[dict], sentences: list[dict]) -> Interval:
    """Shorten a span from the end, landing on the boundary nearest target_len.

    Overshooting target_len is allowed up to max_len, because finishing the
    sentence is worth a few extra seconds and chopping it mid-clause is not.
    The search window is symmetric around the target: as far under as the
    configured headroom lets us go over.
    """
    start, end = span
    if end - start <= target_len:
        return span
    aim = start + target_len
    ceiling = min(start + max_len, end)

    def nearest(source: list[dict]) -> float | None:
        ends = [b["end"] for b in source if start < b["end"] <= ceiling]
        return min(ends, key=lambda b: abs(b - aim)) if ends else None

    # A sentence end is worth a detour, but not an arbitrarily long one: losing
    # eight seconds to finish a clause is a worse clip than ending mid-thought.
    window = max(max_len - target_len, 1.0)
    sentence_end = nearest(sentences)
    if sentence_end is not None and abs(sentence_end - aim) <= window:
        return (start, sentence_end)

    word_end = nearest(words)
    if word_end is not None:
        return (start, word_end)
    return (start, min(aim, end))


# -- assembly ----------------------------------------------------------------

def _assemble(pieces: list[Interval], target: float, max_total: float,
              min_piece: float, words: list[dict],
              sentences: list[dict]) -> tuple[list[Interval], float]:
    """Take pieces in order until the target is met, trimming the last one."""
    chosen: list[Interval] = []
    used = 0.0
    for piece in pieces:
        if used >= target - 1e-3:
            break
        candidate = _trim_end_to(piece, target - used, max_total - used, words, sentences)
        length = candidate[1] - candidate[0]
        if length < min_piece - 1e-3:
            continue
        if used + length > max_total + 1e-3:
            continue
        chosen.append(candidate)
        used += length
    return chosen, used


def _score(chosen: list[Interval], used: float, cfg: Config) -> float:
    # Seconds past the target still count, but at a discount: the headroom is
    # there to finish a thought, not to make every clip as long as allowed.
    target = float(cfg.get("select.target_seconds", 20.0))
    value = min(used, target) + 0.25 * max(0.0, used - target)
    joins = max(0, len(chosen) - 1)
    join_pen = float(cfg.get("select.join_penalty_seconds", 1.5)) * joins
    gap = sum(chosen[i + 1][0] - chosen[i][1] for i in range(len(chosen) - 1))
    gap_pen = float(cfg.get("select.gap_penalty_per_min", 0.5)) * (gap / 60.0)
    return value - join_pen - gap_pen


def _choose(cfg: Config, candidates: list[Interval], clean: list[Interval],
            words: list[dict], sentences: list[dict]) -> dict[str, Any]:
    """Snap, then assemble the best clip out of a candidate set.

    Knows nothing about where the candidates came from, so the same logic serves
    both the picture-and-sound pass and the audio-only fallback.
    """
    target = float(cfg.get("select.target_seconds", 20.0))
    max_total = max(float(cfg.get("select.max_seconds", target)), target)
    min_total = float(cfg.get("select.min_seconds", 8.0))
    min_piece = float(cfg.get("select.min_piece_seconds", 3.0))
    max_joins = int(cfg.get("select.max_joins", 2))
    pool_size = int(cfg.get("select.candidate_pool", 10))

    snapped: list[Interval] = []
    for span in candidates:
        s = _snap_span(
            span, words, sentences,
            bool(cfg.get("select.snap_to_sentence", True)),
            float(cfg.get("select.head_pad", 0.1)),
            float(cfg.get("select.tail_pad", 0.25)),
            clean,
        )
        if s and s[1] - s[0] >= min_piece:
            snapped.append(s)
    snapped = normalize(snapped)

    out: dict[str, Any] = {
        "num_candidates": len(snapped),
        "candidate_total": sum(e - s for s, e in snapped),
    }
    if not snapped:
        out.update(status="failed", reason="no candidate segment survived filtering",
                   pieces=[], total=0.0, joins=0)
        return out

    # 1. a single continuous stretch is always preferred
    longest = max(snapped, key=lambda p: p[1] - p[0])
    if longest[1] - longest[0] >= target:
        pieces, used = _assemble([longest], target, max_total, min_piece, words, sentences)
        out.update(status="ok", strategy="single", pieces=[list(p) for p in pieces],
                   total=used, joins=0, score=_score(pieces, used, cfg))
        return out

    # 2. otherwise search small chronological combinations
    pool = sorted(snapped, key=lambda p: p[1] - p[0], reverse=True)[:pool_size]
    pool.sort()
    best: tuple[float, list[Interval], float] | None = None
    for size in range(1, max_joins + 2):
        for combo in combinations(pool, size):
            pieces, used = _assemble(list(combo), target, max_total, min_piece, words, sentences)
            if not pieces:
                continue
            score = _score(pieces, used, cfg)
            if best is None or score > best[0]:
                best = (score, pieces, used)

    if best is None or best[2] < min_total:
        got = best[2] if best else 0.0
        out.update(status="failed", pieces=[], total=got, joins=0,
                   reason=f"best assembly only {got:.1f}s < min_seconds {min_total:.1f}s")
        return out

    score, pieces, used = best
    out.update(status="ok", strategy="spliced" if len(pieces) > 1 else "single",
               pieces=[list(p) for p in pieces], total=used,
               joins=len(pieces) - 1, score=score)
    return out


def select_one(cfg: Config, video_id: str, force: bool = False,
               audio_only: bool = False) -> dict[str, Any]:
    cached = None if force else read_stage(cfg.meta_dir, video_id, STAGE)
    if cached:
        LOG.info("[%s] selection cached", video_id)
        return cached

    diar = require_stage(cfg.meta_dir, video_id, "diarize")
    asr = require_stage(cfg.meta_dir, video_id, "asr")
    clean = normalize([tuple(s) for s in diar["clean_spans"]])
    words, sentences = asr["words"], asr["sentences"]

    if audio_only:
        # No picture in the source at all, so this is the intended output rather
        # than a fallback from a failed one.
        record = {"video_id": video_id, "mode": "audio"}
        record.update(_choose(cfg, clean, clean, words, sentences))
        write_stage(cfg.meta_dir, video_id, STAGE, record)
        if record["status"] == "ok":
            LOG.info("[%s] audio: %d piece(s), %.1fs, %d join(s)", video_id,
                     len(record["pieces"]), record["total"], record["joins"])
        else:
            LOG.warning("[%s] selection failed: %s", video_id, record["reason"])
        return record

    asd = require_stage(cfg.meta_dir, video_id, "asd")
    good = normalize([tuple(s) for s in asd["good_spans"]])

    record = {"video_id": video_id, "mode": "video"}
    record.update(_choose(cfg, intersect(clean, good), clean, words, sentences))

    if record["status"] == "ok":
        write_stage(cfg.meta_dir, video_id, STAGE, record)
        LOG.info("[%s] %d piece(s), %.1fs total, %d join(s)",
                 video_id, len(record["pieces"]), record["total"], record["joins"])
        return record

    video_reason = record["reason"]
    LOG.warning("[%s] no usable picture: %s", video_id, video_reason)

    # The picture failed, but the audio may still be a clean solo stretch, and a
    # voice clip is worth more than nothing. The speaker rules are unchanged -
    # `clean` already excludes every other voice - only the on-camera test is
    # dropped, so this must be labelled audio and never mixed in with the video
    # deliverables.
    if not bool(cfg.get("select.audio_fallback", True)):
        write_stage(cfg.meta_dir, video_id, STAGE, record)
        return record

    audio_record = {"video_id": video_id, "mode": "audio"}
    audio_record.update(_choose(cfg, clean, clean, words, sentences))
    if audio_record["status"] == "ok":
        audio_record["reason"] = f"picture unusable ({video_reason}); audio only"
        write_stage(cfg.meta_dir, video_id, STAGE, audio_record)
        LOG.info("[%s] audio-only fallback: %d piece(s), %.1fs, %d join(s)",
                 video_id, len(audio_record["pieces"]), audio_record["total"],
                 audio_record["joins"])
        return audio_record

    record["reason"] = f"{video_reason}; audio fallback also failed"
    write_stage(cfg.meta_dir, video_id, STAGE, record)
    LOG.warning("[%s] selection failed: %s", video_id, record["reason"])
    return record


def select_batch(cfg: Config, video_ids: list[str], force: bool = False) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for vid in video_ids:
        try:
            results[vid] = select_one(cfg, vid, force=force)
        except Exception as exc:
            LOG.error("[%s] selection failed: %s", vid, exc)
            results[vid] = {"error": str(exc)}
    return results
