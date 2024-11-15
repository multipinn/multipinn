import numpy as np
import pytest

from multipinn.geometry.domain_nd import *


@pytest.fixture
def hypercube():
    return Hypercube(low=[0, 0], high=[1, 1])


@pytest.fixture
def hypersphere():
    return Hypersphere(center=[0, 0], radius=1)


def test_hypercube_initialization():
    with pytest.raises(ValueError):
        Hypercube(low=[0], high=[1, 1])
    with pytest.raises(ValueError):
        Hypercube(low=[1, 0], high=[0, 1])


def test_hypercube_inside(hypercube):
    assert np.array_equal(
        hypercube.inside(np.array([[0.5, 0.5], [1.5, 0.5]])), [True, False]
    )


def test_hypercube_on_boundary(hypercube):
    assert np.array_equal(
        hypercube.on_boundary(np.array([[0, 0], [1, 1], [0.5, 0.5]])),
        [True, True, False],
    )


def test_hypercube_closest_point(hypercube):
    assert np.array_equal(
        hypercube.closest_point(np.array([[1.5, -0.5], [0.5, 0.5]])),
        [[1, 0], [0.5, 0.5]],
    )


def test_hypersphere_inside(hypersphere):
    assert np.array_equal(
        hypersphere.inside(np.array([[0, 0], [1.5, 0], [0, 1]])), [True, False, True]
    )


def test_hypersphere_on_boundary(hypersphere):
    assert np.array_equal(
        hypersphere.on_boundary(np.array([[0, 1], [1, 0], [2, 0]])), [True, True, False]
    )


def test_hypersphere_closest_point(hypersphere):
    assert np.array_equal(
        hypersphere.closest_point(np.array([[1.5, 0], [0, 2]])), [[1, 0], [0, 1]]
    )


def test_hypercube_random_points(hypercube):
    points = hypercube.random_points(10)
    assert points.shape == (10, 2)
    assert np.all(points >= hypercube.low) and np.all(points <= hypercube.high)


def test_hypersphere_random_points(hypersphere):
    points = hypersphere.random_points(10)
    assert points.shape == (10, 2)
    assert np.all(
        np.linalg.norm(points - hypersphere.center, axis=1) <= hypersphere.radius
    )


@pytest.mark.parametrize("n, expected_shape", [(16, (16, 2)), (9, (9, 2))])
def test_hypercube_uniform_points(hypercube, n, expected_shape):
    points = hypercube.uniform_points(n)
    assert points.shape == expected_shape


@pytest.mark.parametrize("n, expected_shape", [(10, (10, 2)), (5, (5, 2))])
def test_hypersphere_random_boundary_points(hypersphere, n, expected_shape):
    boundary_points = hypersphere.random_boundary_points(n)
    assert boundary_points.shape == expected_shape
