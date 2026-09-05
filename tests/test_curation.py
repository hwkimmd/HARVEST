import numpy as np

from harvest.curation import _sqrt_quota


def test_square_root_quota_is_exact_deterministic_and_capped():
    ids = [9, 2, 7]
    capacities = np.array([100, 400, 900])
    first = _sqrt_quota(ids, capacities, 600)
    second = _sqrt_quota(ids, capacities, 600)
    assert np.array_equal(first, second)
    assert first.sum() == 600
    assert np.all(first <= capacities)
    assert first[2] > first[1] > first[0]
