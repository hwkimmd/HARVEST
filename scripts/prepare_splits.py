#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from harvest.splits import build_assignments


def main():
    parser = argparse.ArgumentParser(
        description="Create the fixed HARVEST test/development folds"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    rows = build_assignments(pd.read_csv(args.manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output, index=False)
    print(rows.groupby(["partition", "cv_fold", "center", "label"]).size())


if __name__ == "__main__":
    main()
