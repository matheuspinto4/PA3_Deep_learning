"""Pré-redimensiona train, val e test_public para o cache em `cache/` (roda uma vez)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pa3.data import SPLITS, build_cache


def main():
    for split in SPLITS:
        t = time.time()
        build_cache(split)
        print(f"{split}: ok ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
