
import pandas as pd
from harvest.splits import build_assignments


def test_fixed_test_and_dev_folds_are_disjoint_and_stratified():
    rows = []
    for center in ("center_1", "center_2"):
        for label in (0, 1):
            for index in range(40):
                rows.append({"image_id": f"{center}_{label}_{index}", "relative_path": f"dummy/{center}_{label}_{index}.png", "label": label, "center": center})
    assigned = build_assignments(pd.DataFrame(rows))
    assert set(assigned.partition) == {"development", "test"}
    assert set(assigned.loc[assigned.partition == "development", "cv_fold"]) == set(range(5))
    assert (assigned.loc[assigned.partition == "test", "cv_fold"] == -1).all()
    for fold in range(5):
        validation = assigned[(assigned.partition == "development") & (assigned.cv_fold == fold)]
        assert set(zip(validation.center, validation.label)) == {("center_1", 0), ("center_1", 1), ("center_2", 0), ("center_2", 1)}
