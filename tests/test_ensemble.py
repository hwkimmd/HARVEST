import numpy as np

from harvest.ensemble import COMPONENTS, aggregate_harvest, middle_three_mean, stable_sigmoid


def test_middle_three_rejects_extremes():
    values = np.array([[100.0], [1.0], [2.0], [3.0], [-100.0]])
    assert middle_three_mean(values)[0] == 2.0


def test_complete_dual_component_ensemble():
    first = np.arange(20, dtype=float).reshape(5, 4)
    second = first + 10.0
    components = {
        COMPONENTS[0]: (first, [0] * 5, [1] * 5),
        COMPONENTS[1]: (second, [10] * 5, [1] * 5),
    }
    scores = aggregate_harvest(components)
    assert scores.shape == (4,)
    assert np.all((scores >= 0) & (scores <= 1))
    assert np.isfinite(stable_sigmoid(np.array([-1000.0, 1000.0]))).all()


def test_missing_component_is_rejected():
    try:
        aggregate_harvest({COMPONENTS[0]: (np.zeros((5, 1)), np.zeros(5), np.ones(5))})
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete HARVEST ensemble was accepted")
