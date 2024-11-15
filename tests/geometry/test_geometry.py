import numpy as np
import pytest

from multipinn.geometry import Hypercube
from multipinn.geometry.geometry import *


def test_isclose():
    # Test values that are close
    assert isclose(1.0, 1.0)
    assert isclose(1.000001, 1.0)

    # Test values that are not close
    assert not isclose(1.0, 1.1)
    assert not isclose(1.0, 2.0)

    # Edge case: very small differences
    assert isclose(1.0, 1.0 + 1e-7)


@pytest.fixture
def simple_geometry():
    return Hypercube([0, 0], [1, 1])


def test_geometry_initialization(simple_geometry):
    assert simple_geometry.dim == 2
    assert np.array_equal(simple_geometry.bbox, np.array([[0, 0], [1, 1]]))
    assert np.isclose(simple_geometry.diam, 1.4142135)


def test_random_points(simple_geometry):
    points = simple_geometry.random_points(10)
    assert points.shape == (10, 2)

    for point in points:
        assert np.all(point >= simple_geometry.bbox[0])
        assert np.all(point <= simple_geometry.bbox[1])
