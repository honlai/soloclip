"""Stage 4: word-level transcription with faster-whisper.

We do not care about the transcript itself - we need word boundaries so that
cut points never land in the middle of a word, and sentence boundaries so that
a spliced clip still sounds like finished thoughts.
"""

from __future__ import annotations

from typing import Any

from .config import Config
from .utils import LOG, free_gpu, load_path, read_stage, require_stage, write_stage

STAGE = "asr"

SENTENCE_END = set(".!?。！？…")


def _load_model(cfg: Config):
    from faster_whisper import WhisperModel

    name = str(cfg.get("asr.model", "small"))
    device = cfg.device
    compute = str(cfg.get("asr.compute_type", "int8_float16"))
    if device == "cpu" and "float16" in compute:
        compute = "int8"  # float16 compute is not available on CPU
    LOG.info("loading faster-whisper %s (%s/%s)", name, device, compute)
    return WhisperModel(name, device=device, compute_type=compute)


def transcribe_one(cfg: Config, video_id: str, force: bool = False, model=None) -> dict[str, Any]:
    cached = None if force else read_stage(cfg.meta_dir, video_id, STAGE)
    if cached:
        LOG.info("[%s] transcript cached", video_id)
        return cached

    audio = require_stage(cfg.meta_dir, video_id, "audio")
    own_model = model is None
    model = model or _load_model(cfg)

    LOG.info("[%s] transcribing", video_id)
    segments, info = model.transcribe(
        str(load_path(cfg.data_root, audio["wav_path"])),
        language=cfg.get("asr.language") or None,
        word_timestamps=bool(cfg.get("asr.word_timestamps", True)),
        vad_filter=False,  # diarization already gives us the speech mask
    )

    words: list[dict[str, Any]] = []
    sentences: list[dict[str, Any]] = []
    for seg in segments:
        seg_words = list(seg.words or [])
        for w in seg_words:
            words.append({"start": float(w.start), "end": float(w.end), "word": w.word})
        sentences.extend(_split_sentences(seg_words, seg.text))

    record = {
        "video_id": video_id,
        "language": getattr(info, "language", None),
        "words": words,
        "sentences": sentences,
        "num_words": len(words),
    }
    write_stage(cfg.meta_dir, video_id, STAGE, record)
    LOG.info("[%s] %d words, %d sentences (lang=%s)",
             video_id, len(words), len(sentences), record["language"])
    if own_model:
        del model
        free_gpu()
    return record


def _split_sentences(seg_words: list[Any], fallback_text: str) -> list[dict[str, Any]]:
    """Group a whisper segment's words into sentences on terminal punctuation."""
    if not seg_words:
        return []
    out: list[dict[str, Any]] = []
    buf: list[Any] = []
    for w in seg_words:
        buf.append(w)
        if w.word.strip()[-1:] in SENTENCE_END:
            out.append(_sentence(buf))
            buf = []
    if buf:
        out.append(_sentence(buf))
    if not out:
        out = [{"start": float(seg_words[0].start), "end": float(seg_words[-1].end),
                "text": fallback_text.strip()}]
    return out


def _sentence(buf: list[Any]) -> dict[str, Any]:
    return {
        "start": float(buf[0].start),
        "end": float(buf[-1].end),
        "text": "".join(w.word for w in buf).strip(),
    }


def transcribe_batch(cfg: Config, video_ids: list[str], force: bool = False) -> dict[str, Any]:
    results: dict[str, Any] = {}
    pending = [v for v in video_ids if force or read_stage(cfg.meta_dir, v, STAGE) is None]
    model = _load_model(cfg) if pending else None
    try:
        for vid in video_ids:
            try:
                results[vid] = transcribe_one(cfg, vid, force=force, model=model)
            except Exception as exc:
                LOG.error("[%s] transcription failed: %s", vid, exc)
                results[vid] = {"error": str(exc)}
    finally:
        del model
        free_gpu()
    return results
