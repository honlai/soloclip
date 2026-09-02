"""Identify the recurring host across a list of interviews.

No manual labelling is needed, because the host gives themselves away
structurally: they appear in *every* episode while each guest appears in only
one. So the voice whose embedding recurs across the most distinct videos is the
host. Anything that shows up in a single video cannot be.

The profile is only advisory - selection still prefers a guest but falls back to
the host rather than producing nothing.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .config import Config
from .utils import LOG, read_stage

PROFILE = "host_profile.json"


def profile_path(cfg: Config):
    return cfg.meta_dir / PROFILE


def load_profile(cfg: Config) -> dict[str, Any] | None:
    path = profile_path(cfg)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOG.warning("corrupt host profile, ignoring: %s", path)
        return None
    return data if data.get("centroid") else None


def _collect(cfg: Config, video_ids: list[str]) -> tuple[np.ndarray, list[tuple[str, str, float]]]:
    """Every (video, speaker) embedding we have on disk, as unit vectors."""
    vectors: list[np.ndarray] = []
    meta: list[tuple[str, str, float]] = []
    for vid in video_ids:
        rec = read_stage(cfg.meta_dir, vid, "diarize")
        if not rec:
            continue
        totals = rec.get("speaker_totals", {})
        for spk, vec in (rec.get("embeddings") or {}).items():
            arr = np.asarray(vec, dtype=np.float32)
            if arr.size == 0 or not np.isfinite(arr).all():
                continue
            vectors.append(arr)
            meta.append((vid, spk, float(totals.get(spk, 0.0))))
    return (np.stack(vectors) if vectors else np.empty((0, 0), np.float32)), meta


def build_profile(cfg: Config, video_ids: list[str]) -> dict[str, Any] | None:
    """Find the voice present in the most distinct videos."""
    vectors, meta = _collect(cfg, video_ids)
    if len(vectors) < 2:
        LOG.warning("not enough speaker embeddings to identify a host (%d)", len(vectors))
        return None

    threshold = float(cfg.get("host.distance", 0.55))
    min_videos = int(cfg.get("host.min_videos", 3))
    sims = vectors @ vectors.T
    near = sims >= (1.0 - threshold)

    # score each candidate by how many *distinct* videos it turns up in, not how
    # many rows match: a talkative guest split into several speakers must not
    # outrank a host who appears once per episode
    best_index, best_videos = -1, 0
    for i in range(len(vectors)):
        videos = {meta[j][0] for j in np.flatnonzero(near[i])}
        if len(videos) > best_videos:
            best_index, best_videos = i, len(videos)

    total_videos = len({m[0] for m in meta})
    if best_videos < min_videos:
        LOG.info("no recurring voice across >=%d videos (best %d of %d) - no host profile",
                 min_videos, best_videos, total_videos)
        return None

    members = np.flatnonzero(near[best_index])
    centroid = vectors[members].mean(axis=0)
    centroid /= float(np.linalg.norm(centroid)) or 1.0
    profile = {
        "centroid": centroid.astype(float).tolist(),
        "videos": best_videos,
        "total_videos": total_videos,
        "coverage": round(best_videos / max(total_videos, 1), 3),
        "distance": threshold,
        "speech_seconds": round(sum(meta[j][2] for j in members), 1),
        "examples": [f"{meta[j][0]}/{meta[j][1]}" for j in members[:8]],
    }
    profile_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    profile_path(cfg).write_text(json.dumps(profile, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    LOG.info("host profile: one voice in %d/%d videos (%.0f%%), %.0fs of speech",
             best_videos, total_videos, 100 * profile["coverage"], profile["speech_seconds"])
    return profile
