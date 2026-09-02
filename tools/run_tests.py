"""Run the unit tests without requiring pytest in the environment."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "src")]

import test_intervals  # noqa: E402
import test_select  # noqa: E402


def main() -> int:
    failures = 0
    total = 0
    for module in (test_intervals, test_select):
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            total += 1
            try:
                getattr(module, name)()
            except Exception:
                failures += 1
                print(f"FAIL {module.__name__}.{name}")
                traceback.print_exc()
    print(f"{total - failures}/{total} passed" if failures else f"all {total} tests pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
