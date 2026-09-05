#!/usr/bin/env python3
"""Build the deterministic GastroNet-5M index used by HARVEST."""
from __future__ import annotations

import argparse
from pathlib import Path

from harvest.curation import scan_gastronet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    index = scan_gastronet(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(args.output, index=False)
    print(f"indexed {len(index):,} images -> {args.output}")


if __name__ == "__main__":
    main()
