"""Normalise an image folder to .jpg, dispatching on real content, not extension.

Extensions in downloaded photo sets lie: this set had .jfif files that were
already JPEG, a .png that was actually WebP, and .jpeg files that were really
PNG. Trusting the suffix would re-encode files that needed only a rename and
skip files that needed real conversion, so every file is sniffed first.

A file that is already JPEG is renamed, never re-encoded - re-encoding JPEG is
lossy for no benefit. Originals are removed only after the replacement has been
opened and verified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

QUALITY = 95


def unique(path: Path) -> Path:
    """A free filename, so a rename never clobbers an unrelated image."""
    if not path.exists():
        return path
    for n in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"no free name for {path}")


def convert(src: Path) -> tuple[str, Path | None]:
    try:
        with Image.open(src) as im:
            fmt = (im.format or "").upper()
            size = im.size
    except Exception as exc:
        return f"unreadable ({type(exc).__name__})", None

    if fmt == "JPEG":
        if src.suffix.lower() == ".jpg":
            return "already jpg", src
        target = unique(src.with_suffix(".jpg"))
        src.rename(target)          # same bytes, no re-encode
        return "renamed", target

    target = unique(src.with_suffix(".jpg"))
    try:
        with Image.open(src) as im:
            im.seek(0)              # animated webp/gif: keep the first frame
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                flat = Image.new("RGB", im.size, (255, 255, 255))
                flat.paste(im, mask=im.split()[-1])
                im = flat
            else:
                im = im.convert("RGB")
            im.save(target, "JPEG", quality=QUALITY, optimize=True)
        with Image.open(target) as check:   # verify before destroying the source
            if check.size != size:
                raise ValueError(f"size changed {size} -> {check.size}")
            check.load()
    except Exception as exc:
        target.unlink(missing_ok=True)
        return f"FAILED ({type(exc).__name__}: {exc})", None

    src.unlink()
    return f"converted from {fmt}", target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.folder)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 1

    files = sorted(p for p in root.rglob("*") if p.is_file())
    counts: dict[str, int] = {}
    failures: list[str] = []
    for path in files:
        if args.dry_run:
            try:
                with Image.open(path) as im:
                    action = f"would handle {im.format}"
            except Exception as exc:
                action = f"unreadable ({type(exc).__name__})"
        else:
            action, _ = convert(path)
        key = action.split(" (")[0]
        counts[key] = counts.get(key, 0) + 1
        if action.startswith("FAILED") or action.startswith("unreadable"):
            failures.append(f"{path}: {action}")

    for key in sorted(counts):
        print(f"  {counts[key]:>3}  {key}")
    for line in failures:
        print(f"  !! {line}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
