import numpy as np
import pytest

from adaptive_mm.estimators import ewma


def test_ewma_hand_calculated_sequence():
    values = [1, -1, 1]
    estimates = ewma(values, half_life=1)

    assert np.allclose(estimates, [1, -1 / 3, 3 / 7])


def test_ewma_matches_direct_weighted_average():
    values = np.array([0.25, -1.5, 2.0, 0.75, -0.5])
    half_life = 2.5
    decay = 2 ** (-1 / half_life)
    expected = []

    for i in range(len(values)):
        weights = decay ** np.arange(i, -1, -1)
        expected.append(np.sum(weights * values[: i + 1]) / np.sum(weights))

    assert np.allclose(ewma(values, half_life), expected)


def test_ewma_constant_sequence_and_output_length():
    values = np.full(6, -0.4)
    estimates = ewma(values, half_life=3)

    assert np.allclose(estimates, values)
    assert len(estimates) == len(values)


@pytest.mark.parametrize("half_life", [0, -1])
def test_ewma_rejects_nonpositive_half_life(half_life):
    with pytest.raises(ValueError):
        ewma([1, -1], half_life)


def test_ewma_rejects_non_one_dimensional_values():
    with pytest.raises(ValueError):
        ewma([[1, -1], [-1, 1]], half_life=1)
