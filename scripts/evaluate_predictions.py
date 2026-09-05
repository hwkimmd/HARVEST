#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from harvest.metrics import evaluate_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed label and score columns with the RARE26 metric"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    rows = pd.read_csv(args.predictions)
    missing = {"label", "score"} - set(rows.columns)
    if missing:
        raise ValueError(f"prediction CSV missing columns: {sorted(missing)}")
    result = evaluate_predictions(rows["label"], rows["score"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
