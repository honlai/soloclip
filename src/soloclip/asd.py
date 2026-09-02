"""Stage 5: active-speaker detection on the video track.

For every sampled frame we ask three questions:
  1. is there a big enough face, roughly facing the camera?
  2. is it the *same* person as the dominant speaker (face embedding)?
  3. are the lips actually moving?

(3) is measured on the identity-aligned 112x112 crop, so head motion is largely
cancelled out and what remains in the mouth ROI is mouth motion.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import Config
from .intervals import normalize, runs_from_flags
from .utils import (LOG, free_gpu, load_path, read_stage, require_stage,
                    write_stage)

STAGE = "asd"

DETECT_HEIGHT = 480      # frames are piped at this height; enough for face detection
ALIGNED_SIZE = 112       # insightface norm_crop output
MOUTH_ROI = (72, 112, 28, 84)  # y0, y1, x0, x1 within the aligned crop


# -- frame source ------------------------------------------------------------

def _frame_stream(video: Path, fps: float, seconds: float, width: int, height: int) -> Iterator[np.ndarray]:
    """Yield BGR frames at a fixed rate, so frame i is exactly at t = i / fps."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video),
        "-t", f"{seconds:.3f}",
        "-vf", f"fps={fps},scale={width}:{height}",
        "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]
    LOG.debug("exec: %s", " ".join(cmd))
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        proc.wait()
        if proc.returncode not in (0, None) and err.strip():
            LOG.warning("ffmpeg frame reader: %s", err.strip().splitlines()[-1])


def _scaled_size(width: int, height: int) -> tuple[int, int]:
    if height <= DETECT_HEIGHT or height == 0:
        return (width - width % 2, height - height % 2)
    target_h = DETECT_HEIGHT
    target_w = int(round(width * target_h / height))
    return (target_w - target_w % 2, target_h - target_h % 2)


# -- face model --------------------------------------------------------------

def _load_face_app(cfg: Config):
    from insightface.app import FaceAnalysis

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if cfg.device == "cuda" else ["CPUExecutionProvider"])
    app = FaceAnalysis(
        name="buffalo_l",
        # genderage / 2d106 are dead weight here; pose comes from landmark_3d_68
        allowed_modules=["detection", "landmark_3d_68", "recognition"],
        providers=providers,
    )
    app.prepare(ctx_id=0 if cfg.device == "cuda" else -1, det_size=(640, 640))
    return app


def _pose(face) -> tuple[float, float]:
    """Return (yaw, pitch) in degrees, falling back to a keypoint proxy."""
    pose = getattr(face, "pose", None)
    if pose is not None and len(pose) >= 2:
        pitch, yaw = float(pose[0]), float(pose[1])
        return yaw, pitch
    kps = np.asarray(face.kps, dtype=np.float32)
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_mid = (left_eye + right_eye) / 2.0
    eye_dist = float(np.linalg.norm(right_eye - left_eye)) or 1.0
    ratio = float(nose[0] - eye_mid[0]) / eye_dist
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * ratio))))
    return yaw, 0.0


def _mouth_patch(frame: np.ndarray, face) -> np.ndarray | None:
    """Identity-aligned mouth crop; alignment removes most head-motion energy."""
    from insightface.utils import face_align

    try:
        aligned = face_align.norm_crop(frame, landmark=face.kps, image_size=ALIGNED_SIZE)
    except Exception:
        return None
    y0, y1, x0, x1 = MOUTH_ROI
    patch = aligned[y0:y1, x0:x1]
    return patch.mean(axis=2).astype(np.float32) / 255.0


# -- identity reference ------------------------------------------------------

def _reference_embedding(embeddings: np.ndarray, threshold: float, cap: int = 1500) -> np.ndarray:
    """Centroid of the largest tight cluster - i.e. the most-present face."""
    sample = embeddings
    if len(sample) > cap:
        idx = np.linspace(0, len(sample) - 1, cap).astype(int)
        sample = sample[idx]
    sims = sample @ sample.T                      # rows are unit vectors
    neighbours = sims >= (1.0 - threshold)
    seed = int(neighbours.sum(axis=1).argmax())
    members = sample[neighbours[seed]]
    ref = members.mean(axis=0)
    return ref / (np.linalg.norm(ref) or 1.0)


def _smooth(flags: np.ndarray, window: int) -> np.ndarray:
    """Median filter so one missed detection does not shatter a good run."""
    if window < 3 or len(flags) < window:
        return flags
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(flags.astype(np.uint8), pad, mode="edge")
    stacked = np.stack([padded[i:i + len(flags)] for i in range(window)])
    return np.median(stacked, axis=0) > 0.5


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window < 2 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


# -- per-frame measurements --------------------------------------------------

FRAME_KEYS = ("times", "detected", "face_ratio", "yaw_deg", "pitch_deg", "id_dist", "lip_energy")


def _frames_path(cfg: Config, video_id: str) -> Path:
    return cfg.cache_dir / f"{video_id}.asd_frames.npz"


def _save_frames(cfg: Config, video_id: str, frames: dict[str, Any]) -> None:
    """Keep the raw per-frame measurements so thresholds can be retuned without
    paying for face detection again - the expensive part is measuring, not judging."""
    path = _frames_path(cfg, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **frames)


def load_frames(cfg: Config, video_id: str) -> dict[str, Any] | None:
    path = _frames_path(cfg, video_id)
    if not path.exists():
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def score_frames(cfg: Config, video_id: str, frames: dict[str, Any]) -> dict[str, Any]:
    """Apply the configured thresholds to stored measurements. Pure CPU, instant."""
    times = frames["times"]
    detected = frames["detected"].astype(bool)
    yaw_arr, pitch_arr = frames["yaw_deg"], frames["pitch_deg"]
    id_dist, lip_energy = frames["id_dist"], frames["lip_energy"]
    step = float(frames["step"])
    fps = float(frames["fps"])

    size_ok = frames["face_ratio"] >= float(cfg.get("asd.min_face_ratio", 0.06))
    # A camera above or below eye level puts a constant offset on pitch, so the
    # two axes get their own thresholds instead of one folded-together number.
    yaw_ok = yaw_arr <= float(cfg.get("asd.max_yaw_deg", 30))
    pitch_ok = pitch_arr <= float(cfg.get("asd.max_pitch_deg", 25))
    id_ok = id_dist <= float(cfg.get("asd.face_id_threshold", 0.35))
    pass_flags = detected & size_ok & yaw_ok & pitch_ok & id_ok
    lip_flags = lip_energy >= float(cfg.get("asd.lip_motion_min", 0.015))

    good = _smooth(pass_flags & lip_flags, window=3)
    spans = runs_from_flags(list(times), list(good), step)

    # a smoothed span still has to be mostly genuine detections
    min_ratio = float(cfg.get("asd.min_frontal_ratio", 0.8))
    kept: list[tuple[float, float]] = []
    scores: list[dict[str, Any]] = []
    for s, e in spans:
        mask = (times >= s) & (times < e)
        if not mask.any():
            continue
        ratio = float(pass_flags[mask].mean())
        if ratio < min_ratio:
            continue
        kept.append((s, e))
        scores.append({
            "start": s, "end": e,
            "frontal_ratio": ratio,
            "lip_energy": float(lip_energy[mask].mean()),
            "id_dist": float(id_dist[mask & detected].mean()) if (mask & detected).any() else 1.0,
        })

    # Per-criterion rates, measured only over frames that had a face at all, so a
    # failing run says *which* threshold is doing the rejecting instead of just
    # reporting one opaque pass rate.
    def rate(mask: np.ndarray) -> float:
        return float(mask[detected].mean()) if detected.any() else 0.0

    def pct(values: np.ndarray, qs=(10, 50, 90)) -> dict[str, float]:
        sel = values[detected]
        return {f"p{q}": round(float(np.percentile(sel, q)), 4) for q in qs} if len(sel) else {}

    return {
        "video_id": video_id,
        "fps": fps,
        "frames": len(times),
        "detect_rate": float(detected.mean()),
        "pass_rate": float(pass_flags.mean()),
        "criteria": {
            "face_size_ok": rate(size_ok),
            "yaw_ok": rate(yaw_ok),
            "pitch_ok": rate(pitch_ok),
            "pose_ok": rate(yaw_ok & pitch_ok),
            "identity_ok": rate(id_ok),
            "lip_motion_ok": rate(lip_flags),
            "all_ok": rate(pass_flags & lip_flags),
        },
        "distributions": {
            "face_ratio": pct(frames["face_ratio"]),
            "yaw_deg": pct(yaw_arr),
            "pitch_deg": pct(pitch_arr),
            "id_dist": pct(id_dist),
            "lip_energy": pct(lip_energy),
        },
        "good_spans": kept,
        "span_scores": scores,
    }


def rescore_one(cfg: Config, video_id: str) -> dict[str, Any]:
    """Re-apply thresholds to cached measurements. No GPU, no video decode."""
    frames = load_frames(cfg, video_id)
    if frames is None:
        raise RuntimeError(f"[{video_id}] no cached frame measurements; run `asd` first")
    record = score_frames(cfg, video_id, frames)
    write_stage(cfg.meta_dir, video_id, STAGE, record)
    LOG.info("[%s] rescored: %d spans, %.0f%% of detected frames pass",
             video_id, len(record["good_spans"]), 100 * record["criteria"]["all_ok"])
    return record


# -- stage -------------------------------------------------------------------

def analyse_one(cfg: Config, video_id: str, force: bool = False, app=None) -> dict[str, Any]:
    cached = None if force else read_stage(cfg.meta_dir, video_id, STAGE)
    if cached:
        LOG.info("[%s] asd cached", video_id)
        return cached

    audio = require_stage(cfg.meta_dir, video_id, "audio")
    diar = require_stage(cfg.meta_dir, video_id, "diarize")
    clean = normalize([tuple(s) for s in diar["clean_spans"]])
    if not clean:
        raise RuntimeError(f"[{video_id}] no audio-clean spans to analyse")

    own_app = app is None
    app = app or _load_face_app(cfg)

    fps = float(cfg.get("asd.fps", 6))
    seconds = float(audio["scan_seconds"])
    width, height = _scaled_size(int(audio["width"]), int(audio["height"]))
    step = 1.0 / fps
    n_expected = int(seconds * fps)

    # only frames inside audio-clean spans are worth running the detector on
    def in_clean(t: float) -> bool:
        return any(s <= t < e for s, e in clean)

    times: list[float] = []
    face_ok: list[bool] = []
    yaws: list[float] = []
    pitches: list[float] = []
    sizes: list[float] = []
    embeds: list[np.ndarray] = []
    mouth_diff: list[float] = []
    prev_patch: np.ndarray | None = None
    prev_index = -2

    LOG.info("[%s] sampling %d frames at %gfps (%dx%d)", video_id, n_expected, fps, width, height)
    report_every = max(1, n_expected // 10)
    source = load_path(cfg.data_root, audio["video_path"])
    for i, frame in enumerate(_frame_stream(source, fps, seconds, width, height)):
        t = i * step
        if i and i % report_every == 0:
            LOG.info("[%s]   %d/%d frames (t=%.0fs)", video_id, i, n_expected, t)
        times.append(t)
        if not in_clean(t):
            face_ok.append(False); yaws.append(180.0); pitches.append(180.0); sizes.append(0.0)
            embeds.append(np.zeros(512, np.float32)); mouth_diff.append(0.0)
            prev_patch, prev_index = None, i
            continue

        faces = app.get(frame)
        if not faces:
            face_ok.append(False); yaws.append(180.0); pitches.append(180.0); sizes.append(0.0)
            embeds.append(np.zeros(512, np.float32)); mouth_diff.append(0.0)
            prev_patch, prev_index = None, i
            continue

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        yaw, pitch = _pose(face)
        emb = np.asarray(face.normed_embedding, dtype=np.float32)

        face_ok.append(True)
        yaws.append(abs(yaw))
        pitches.append(abs(pitch))
        sizes.append(float(face.bbox[2] - face.bbox[0]) / width)
        embeds.append(emb)

        patch = _mouth_patch(frame, face)
        if patch is not None and prev_patch is not None and i == prev_index + 1 \
                and patch.shape == prev_patch.shape:
            mouth_diff.append(float(np.abs(patch - prev_patch).mean()))
        else:
            mouth_diff.append(0.0)
        prev_patch, prev_index = patch, i

    if not times:
        raise RuntimeError(f"[{video_id}] decoded no frames")

    detected = np.array(face_ok, dtype=bool)
    if detected.sum() < 2:
        raise RuntimeError(f"[{video_id}] almost no faces detected in clean speech")

    emb_arr = np.stack(embeds)
    ref = _reference_embedding(emb_arr[detected], float(cfg.get("asd.face_id_threshold", 0.35)))

    lip_window = max(2, int(round(fps)))  # ~1s of context
    frames = {
        "times": np.array(times, dtype=np.float32),
        "detected": detected,
        "face_ratio": np.array(sizes, dtype=np.float32),
        "yaw_deg": np.array(yaws, dtype=np.float32),
        "pitch_deg": np.array(pitches, dtype=np.float32),
        "id_dist": (1.0 - (emb_arr @ ref)).astype(np.float32),
        "lip_energy": _rolling_mean(np.array(mouth_diff, dtype=np.float32), lip_window),
        "step": np.float32(step),
        "fps": np.float32(fps),
    }
    _save_frames(cfg, video_id, frames)
    record = score_frames(cfg, video_id, frames)
    write_stage(cfg.meta_dir, video_id, STAGE, record)
    LOG.info("[%s] faces on %.0f%% of frames, %d talking-head spans",
             video_id, 100 * record["detect_rate"], len(record["good_spans"]))
    LOG.info("[%s]   of detected frames: size %.0f%% yaw %.0f%% pitch %.0f%% id %.0f%% lip %.0f%% -> all %.0f%%",
             video_id, *[100 * record["criteria"][k] for k in
                         ("face_size_ok", "yaw_ok", "pitch_ok",
                          "identity_ok", "lip_motion_ok", "all_ok")])
    if own_app:
        del app
        free_gpu()
    return record


def analyse_batch(cfg: Config, video_ids: list[str], force: bool = False) -> dict[str, Any]:
    results: dict[str, Any] = {}
    pending = [v for v in video_ids if force or read_stage(cfg.meta_dir, v, STAGE) is None]
    app = _load_face_app(cfg) if pending else None
    try:
        for vid in video_ids:
            try:
                results[vid] = analyse_one(cfg, vid, force=force, app=app)
            except Exception as exc:
                LOG.error("[%s] asd failed: %s", vid, exc)
                results[vid] = {"error": str(exc)}
    finally:
        del app
        free_gpu()
    return results
