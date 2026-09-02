"""Renumber the images in each subfolder to 01.jpg, 02.jpg, ...

Numbering restarts inside every folder, because the folder name already carries
the identity - a flat global sequence would lose that grouping.

Renaming happens in two passes via temporary names. A single pass would clobber
files whenever a target name is already taken by a *different* image, which is
exactly what happens on a second run or when a source is literally called
"01.jpg".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def renumber(folder: Path, ext: str, width: int, dry_run: bool) -> list[tuple[str, str]]:
    files = sorted((p for p in folder.iterdir() if p.is_file()),
                   key=lambda p: p.name.lower())
    plan = [(p, folder / f"{i:0{width}d}{ext}") for i, p in enumerate(files, 1)]
    if all(src == dst for src, dst in plan):
        return []
    if dry_run:
        return [(s.name, d.name) for s, d in plan if s != d]

    staged = []
    for index, (src, _) in enumerate(plan):
        tmp = folder / f".renumber_tmp_{index}"
        src.rename(tmp)
        staged.append(tmp)
    for tmp, (_, dst) in zip(staged, plan):
        tmp.rename(dst)
    return [(s.name, d.name) for s, d in plan if s.name != d.name]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ext", default=".jpg")
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 1

    folders = sorted(p for p in root.rglob("*") if p.is_dir()
                     and any(c.is_file() for c in p.iterdir()))
    total = 0
    for folder in folders:
        changes = renumber(folder, args.ext, args.width, args.dry_run)
        n = len([c for c in folder.iterdir() if c.is_file()])
        total += n
        verb = "would rename" if args.dry_run else "renamed"
        print(f"  {folder.relative_to(root)}: {n} file(s), {verb} {len(changes)}")
    print(f"  total {total} file(s) in {len(folders)} folder(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
