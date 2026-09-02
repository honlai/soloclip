"""Make stage caches portable, and follow a data directory that moved.

Stage records used to store absolute paths, which quietly defeated
`paths.data_root`: point it somewhere else, or merely rename a directory, and
every cached record still referred to the old location even though the files had
come along. Records are relative now; this converts what is already on disk.

    python tools/relocate.py -c configs/talks.yaml            # dry run
    python tools/relocate.py -c configs/talks.yaml --apply
    python tools/relocate.py -c configs/talks.yaml --apply --from-root /old/path
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soloclip.config import load_config  # noqa: E402

PATH_KEYS = ("video_path", "wav_path", "out_path")


def remap(value: str, mapping: dict[str, str]) -> str:
    """Swap the first path component when a data directory has been renamed."""
    parts = Path(value).parts
    if parts and parts[0] in mapping:
        return str(Path(mapping[parts[0]], *parts[1:]))
    return value


def convert(value: str, data_root: Path, from_root: Path | None) -> str | None:
    """Return the relative form, or None when nothing needs changing."""
    path = Path(value)
    if not path.is_absolute():
        return None
    for root in filter(None, (data_root, from_root)):
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    # Same-named data dir under a different parent, e.g. after a rename: keep
    # the tail that starts at a directory we recognise.
    parts = path.parts
    for i, part in enumerate(parts):
        if (data_root / part).exists():
            return str(Path(*parts[i:]))
    return None


def remap(key: str, value: str, cfg) -> str | None:
    """Point a relative path at the directory the config names now.

    Renaming a data directory leaves stored paths naming the old one. Which
    directory a record refers to is implied by the key, so the leading component
    can simply be replaced rather than guessed at.
    """
    parts = Path(value).parts
    if len(parts) < 2:
        return None
    if key in ("video_path", "wav_path"):
        target = cfg.work / Path(*parts[1:])
    elif key == "out_path":
        base = cfg.audio_out_dir if Path(value).suffix == ".m4a" else cfg.out_dir
        target = base / parts[-1]
    else:
        return None
    new = str(target.relative_to(cfg.data_root))
    return new if new != value else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/base.yaml")
    ap.add_argument("--apply", action="store_true", help="write the changes")
    ap.add_argument("--from-root", help="the data root these records were written under")
    ap.add_argument("--map", action="append", default=[], metavar="OLD=NEW",
                    help="rename a leading path component, e.g. --map work=work_talks. "
                         "Repeatable. Needed after renaming a data directory: the "
                         "stored paths are relative, so they follow a *moved* data "
                         "root by themselves, but not a *renamed* subdirectory.")
    ap.add_argument("--remap", action="store_true",
                    help="also rewrite the leading directory of relative paths to match "
                         "the config, for when a data directory was renamed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from_root = Path(args.from_root).resolve() if args.from_root else None
    files = sorted(cfg.meta_dir.glob("*.json"))
    if not files:
        print(f"no stage records under {cfg.meta_dir}")
        return 1

    mapping = dict(m.split("=", 1) for m in args.map)
    changed = fields = 0
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  skipping unreadable {path.name}")
            continue
        touched = False
        for key in PATH_KEYS:
            value = data.get(key)
            if not isinstance(value, str):
                continue
            new = convert(value, cfg.data_root, from_root)
            if new is None and args.remap:
                new = remap(key, value, cfg)
            if new and new != value:
                data[key] = new
                touched = True
                fields += 1
        if touched:
            changed += 1
            if args.apply:
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                encoding="utf-8")

    verb = "rewrote" if args.apply else "would rewrite"
    print(f"  {verb} {fields} path(s) in {changed}/{len(files)} record(s) under {cfg.meta_dir}")
    if not args.apply and changed:
        print("  (dry run - pass --apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
