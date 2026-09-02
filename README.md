# soloclip

**solo** = one person on screen, one voice in the audio.

Give it a list of video URLs. For each one it cuts **a single ~20 second clip of the
same person facing the camera and speaking, with nobody else audible**. When the
picture does not qualify but the audio is still a clean solo stretch, it falls back
to an audio-only clip of the same length.

```
list  →  face detection  →  voice matching (optional)  →  cut & splice  →  transcode (optional)
```

Measured over three real lists: **355 URLs → 248 clips** (217 video, 31 audio-only),
of which 127 (51%) are a single continuous take with no splice at all.

繁體中文說明：[README.zh-tw.md](README.zh-tw.md)　·　简体中文说明：[README.zh-cn.md](README.zh-cn.md)

---

## The five steps

### 1 · The list

One URL per line in `url_list*.txt`. The config says which files to read:

```yaml
paths:
  url_list:
    - url_list_talks.txt
    - url_list_talks_2.txt     # duplicates across lists are dropped by video id
```

Adding a list does not re-run the old ones: videos that already produced a clip are
skipped, which matters because `runtime.cleanup: all` deletes the source afterwards.

### 2 · Face detection

Frames are sampled inside the audio-clean regions and judged on four things. All
four must hold before a frame counts as "this person is talking to the camera":

| Test | Measure | Setting |
| --- | --- | --- |
| Face is big enough | face width as a fraction of frame width | `asd.min_face_ratio` |
| Facing the camera | head yaw and pitch, thresholded separately | `asd.max_yaw_deg` / `max_pitch_deg` |
| Same person throughout | cosine distance between face embeddings | `asd.face_id_threshold` |
| Actually speaking | frame-to-frame energy in the aligned mouth region | `asd.lip_motion_min` |

The identity reference is not "the biggest face". It is the centre of the largest
embedding cluster found while the target speaker is talking, so an audience member
in shot does not hijack it.

**Per-frame measurements are cached as `.npz`**, so retuning thresholds costs no GPU
time at all:

```bash
soloclip -c configs/talks.yaml asd --rescore        # apply new thresholds, seconds
python tools/sweep.py --yaw 30 35 40 --size 0.06 0.05
```

`sweep.py` reports the clip length and splice count each threshold pair would
actually produce. A higher pass rate that still cannot assemble 20 seconds is not
an improvement, so pass rate alone is the wrong thing to look at.

### 3 · Voice matching (optional)

When one host runs through an entire interview series, guest material should win and
the host should be the last resort.

No manual labelling is needed, because the host gives themselves away structurally:
**they appear in every episode while each guest appears in exactly one**. The voice
whose embedding recurs across the most distinct videos is the host.

```bash
soloclip -c configs/interviews.yaml diarize            # produce voice embeddings
soloclip -c configs/interviews.yaml host               # find the recurring voice
soloclip -c configs/interviews.yaml diarize --retarget # apply it, no GPU
```

Selection then targets the most talkative *non-host*, and only falls back to the host
when the guest has neither a usable close-up nor clean audio. Everything downstream is
unchanged, because it all builds on the same clean-span list.

### 4 · Cut & splice

1. Clean regions = target speaker's speech − everyone else's (widened by `overlap_pad`)
   ∩ regions that passed face detection
2. Prefer **the longest single continuous stretch**; if it is long enough, that is the
   clip, with zero splices
3. Only if it is not, assemble several pieces: each ≥ `min_piece_seconds`, at most
   `max_joins` splices, scored as `length − splice penalty − time-gap penalty`, so
   pieces that sit near each other in the source win
4. Every cut point snaps to a word or sentence boundary from ASR. The headroom between
   `target_seconds` and `max_seconds` exists so a sentence can finish
5. If the total cannot reach `min_seconds`, nothing is written. Producing no clip beats
   producing an unusable one

Video joins are hard cuts, audio joins get a 20 ms fade, and the whole clip gets one
loudness pass.

### 5 · Transcode (optional)

```bash
soloclip -c configs/interviews.yaml pair-audio    # an audio twin of every clip
```

This stream-copies the track out of the finished clip rather than selecting again — a
pair is only useful if both halves are the same moment.

---

## Install

```bash
conda create -n soloclip python=3.11 -y && conda activate soloclip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"                       # gives you the `soloclip` command
conda install -c conda-forge nodejs -y        # YouTube's JS signature challenge needs it

# insightface pulls the CPU build of onnxruntime back in, shadowing the GPU one:
pip uninstall -y onnxruntime && pip install --force-reinstall --no-deps onnxruntime-gpu==1.22.0
```

### HuggingFace token (required)

`pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` are gated models.
**You must accept the terms for each of them on the HuggingFace website** — a token
alone is not enough, and without acceptance you get a 403 `GatedRepoError`. This is
the only step that cannot be automated.

The token is read from `HF_TOKEN`, then `.env`, then `~/.cache/huggingface/token`
(what `huggingface-cli login` leaves behind), so a machine that has already logged in
needs no further setup.

### Known version traps

These combinations are already handled in `pyproject.toml` and in the code.
**Read this table before changing a dependency** — several of these fail silently.

| Symptom | Cause and fix |
| --- | --- |
| Face detection is 10× slower, `libcublasLt.so.13 not found` on stderr | onnxruntime-gpu ≥1.23 is built against CUDA 13 while torch ships CUDA 12, so it **silently** falls back to CPU. Pin `onnxruntime-gpu==1.22.0` |
| CUDA provider disappears after installing insightface | insightface depends on the CPU `onnxruntime`; the two packages overwrite each other. Remove the CPU one, then `--force-reinstall --no-deps onnxruntime-gpu` |
| `hf_hub_download() got an unexpected keyword argument 'use_auth_token'` | pyannote 3.4 still passes the old argument, huggingface_hub 1.0 removed it. Pin `huggingface_hub<1.0` |
| HuggingFace downloads stall at 0 bytes while curl is fine | the `hf-xet` transfer backend. `config.py` sets `HF_HUB_DISABLE_XET=1` on import |
| `UnpicklingError: Weights only load failed` | torch 2.6 changed `torch.load` to `weights_only=True`. `diarize._torch_load_patch()` restores the old behaviour for the duration of the pipeline load only |
| `Requested format is not available`, or only storyboards come back | **A JS runtime and the EJS solver script are both required**; without either you get no real formats at all. Install node and set `download.remote_components: "ejs:github"`. This one cost 41 videos, and it disguises itself as SABR or as "only 180p available" — that 180 is the thumbnail height |
| `Sign in to confirm you're not a bot` | needs cookies from a signed-in session: `download.cookies_from_browser`. Under WSL only Firefox works, because Chrome and Edge encrypt cookies with DPAPI |
| `The page needs to be reloaded` | switch player client: `download.youtube_player_client: "web_safari,android"` |

Cookies, player client and JS runtime must all be applied to **both** the probe and the
download path — `download.probe_video_id()` uses a separate clean `YoutubeDL`, so
configuring only the downloader misses half the calls.

---

## Usage

```bash
soloclip -c configs/talks.yaml run          # whole list, start to finish
soloclip -c configs/talks.yaml status       # per-video stage progress
make up     CFG=configs/talks.yaml          # background run with crash restarts
make status CFG=configs/talks.yaml
```

`-c` is a global option, so it goes **before** the subcommand.

Every stage can be re-run on its own, which is how you debug one awkward video without
redoing the expensive work in front of it:

```bash
soloclip -c configs/talks.yaml asd    --video-id=-ABCDEFGHIJ --force
soloclip -c configs/talks.yaml select --video-id=-ABCDEFGHIJ --force
```

Use the `--video-id=` form when an id starts with `-`, or argparse reads it as a flag.

---

## Configuration

`configs/base.yaml` holds the shared thresholds; each list states only its differences:

```yaml
extends: base.yaml

paths:
  url_list: url_list_interviews.txt
  work_dir: work_interviews
```

| File | Purpose |
| --- | --- |
| `configs/base.yaml` | shared thresholds and behaviour |
| `configs/talks.yaml` | lectures and single-speaker talks |
| `configs/interviews.yaml` | interviews, host de-prioritisation enabled |
| `configs/podcasts.yaml` | audio-only sources, face stages skipped |

Machine-specific values (cookie profile paths, storage locations, interpreter paths)
belong in gitignored override files, never in tracked ones. Each has an `.example`:
`configs/local.yaml`, `local.mk`, `tools/local.env`.

---

## Layout

```
src/soloclip/     download → audio → diarize → asr → asd → select → render
configs/          base.yaml plus one file per list
tools/            supervise/watchdog for long runs, sweep for thresholds, relocate for moves
tests/            pytest, no GPU needed
```

Generated data is kept out of the repository. Where it goes is `paths.data_root`,
which defaults to `".."` — the parent of the clone — so a working directory usually
looks like this:

```
your-working-dir/
├── soloclip/             ← the clone; version control covers only this
│   ├── src/ configs/ tools/ tests/
│   └── README.md LICENSE pyproject.toml Makefile
├── url_list*.txt         ← your lists
├── work*/                ← intermediates (30GB in practice)
├── out*/ out*_audio/     ← results
└── logs*/ var/
```

`SOLOCLIP_DATA` overrides the config, which is how the same config runs on machines
with different storage:

```bash
export SOLOCLIP_DATA=/mnt/bigdisk/soloclip
soloclip -c configs/talks.yaml status
```

Stage records store paths relative to that root, so the data directory can be moved.
Renaming a directory inside it needs one pass of `tools/relocate.py --map old=new`.

`work*/cache/` holds the per-frame face measurements. Deleting it means paying for the
GPU again, so leave it alone before a threshold sweep.

---

## License

MIT, see `LICENSE`.

The code is MIT; **the material it processes is not**. Downloaded videos, the clips cut
from them, and any reference photos remain the copyright of their owners. This tool
does not grant you the right to redistribute them, which is part of why `.gitignore`
keeps all of it out of the repository.
