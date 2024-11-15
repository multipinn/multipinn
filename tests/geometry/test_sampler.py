import numpy as np
import pytest

from multipinn.generation.sampler import (
    _extra_randomization,
    _pseudorandom,
    _quasirandom,
    sample,
)


@pytest.mark.parametrize("n_samples, dimension", [(1, 1), (10, 1), (100, 10), (5, 3)])
def test_pseudorandom_shape(n_samples, dimension):
    result = _pseudorandom(n_samples, dimension)
    assert result.shape == (n_samples, dimension)
    assert result.dtype == np.float32


@pytest.mark.parametrize("n_samples, dimension", [(1, 1), (10, 1), (100, 10), (5, 3)])
@pytest.mark.parametrize("sampler_name", ["LHS", "Halton", "Hammersley", "Sobol"])
def test_quasirandom_shape(n_samples, dimension, sampler_name):
    result = _quasirandom(n_samples, dimension, sampler_name)
    assert result.shape == (n_samples, dimension)
    assert result.dtype == np.float32


@pytest.mark.parametrize("n_samples, dimension", [(1, 1), (10, 1), (100, 10), (5, 3)])
def test_quasirandom_exclusion_boundary_points(n_samples, dimension):
    result = _quasirandom(n_samples, dimension, "Sobol")
    assert not np.any(np.all(result == 0, axis=1))  # No [0, 0, 0, ...]
    assert not np.any(np.all(result == 0.5, axis=1))  # No [0.5, 0.5, 0.5, ...]


@pytest.mark.parametrize("n_samples, dimension", [(1, 1), (10, 1), (100, 10), (5, 3)])
def test_sample_pseudorandom(n_samples, dimension):
    result = sample(n_samples, dimension, "pseudo")
    assert result.shape == (n_samples, dimension)
    assert result.dtype == np.float32


@pytest.mark.parametrize(
    "n_samples, dimension, sampler_name",
    [(1, 1, "LHS"), (10, 1, "Halton"), (100, 10, "Hammersley"), (5, 3, "Sobol")],
)
def test_sample_quasirandom(n_samples, dimension, sampler_name):
    result = sample(n_samples, dimension, sampler_name)
    assert result.shape == (n_samples, dimension)
    assert result.dtype == np.float32


@pytest.mark.parametrize("n_samples", [1, 10, 100, 5])
def test_extra_random_output(n_samples):
    np.random.seed(0)  # For reproducibility
    dimension = 3
    quasirandom_arr = np.random.rand(n_samples, dimension).astype(
        "float32"
    )  # Simulating quasirandom output
    result = _extra_randomization(quasirandom_arr)
    assert result.shape == quasirandom_arr.shape
    assert np.all(result >= 0) and np.all(result < 1)  # Check all values are in [0, 1)


def test_invalid_sampler_name():
    with pytest.raises(ValueError):
        _quasirandom(10, 2, "invalid_sampler")
