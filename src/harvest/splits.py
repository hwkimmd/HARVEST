"""Fixed HARVEST test/development-fold assignment."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 20260830
TEST_FRACTION = 0.20
FOLDS = 5


def build_assignments(rows):
    rows = rows.copy().reset_index(drop=True)
    required = {"image_id", "relative_path", "label", "center"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if rows.image_id.duplicated().any():
        raise ValueError("image_id values must be unique")
    strata = rows.center.astype(str) + "__" + rows.label.astype(int).astype(str)
    development, test = train_test_split(
        np.arange(len(rows)),
        test_size=TEST_FRACTION,
        random_state=SEED,
        shuffle=True,
        stratify=strata,
    )
    rows["partition"] = "test"
    rows["cv_fold"] = -1
    rows.loc[development, "partition"] = "development"
    dev_strata = strata.iloc[development].to_numpy()
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, validation_local) in enumerate(
        splitter.split(development, dev_strata)
    ):
        rows.loc[development[validation_local], "cv_fold"] = fold
    if (rows.loc[rows.partition == "development", "cv_fold"] < 0).any():
        raise RuntimeError("development fold assignment is incomplete")
    return rows
