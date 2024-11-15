import numpy as np
import pytest

from multipinn.geometry.domain_2d import *


@pytest.fixture
def ellipse():
    return Ellipse(center=(0, 0), semimajor=5, semiminor=3, angle=0)


@pytest.fixture
def polygon():
    return Polygon(vertices=[(0, 0), (5, 0), (5, 5), (0, 5)])


def test_ellipse_initialization(ellipse):
    assert ellipse.center.tolist() == [0.0, 0.0]
    assert ellipse.semimajor == 5
    assert ellipse.semiminor == 3
    assert np.isclose(ellipse.focus1, [-4.0, 0.0]).all()
    assert np.isclose(ellipse.focus2, [4.0, 0.0]).all()


def test_ellipse_on_boundary(ellipse):
    assert ellipse.on_boundary(np.array([5.0, 0.0])).all()
    assert not ellipse.on_boundary(np.array([6, 0]))


def test_ellipse_inside(ellipse):
    assert ellipse.inside(np.array([0, 0]))  # Inside the ellipse
    assert not ellipse.inside(np.array([6, 0]))  # Outside the ellipse


def test_ellipse_random_points(ellipse):
    points = ellipse.random_points(1000)
    assert points.shape == (1000, 2)


def test_ellipse_boundary_normal(ellipse):
    normal = ellipse.boundary_normal(np.array([[5, 0]]))
    assert np.isclose(normal.shape, (1, 2)).all()


def test_polygon_initialization(polygon):
    assert polygon.nvertices == 4
    assert polygon.area > 0  # Area should be positive
    assert np.isclose(
        polygon.vertices, np.array([(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)])
    ).all()


def test_polygon_inside(polygon):
    assert polygon.inside(np.array([(1, 1), (3, 3), (6, 6)])).tolist() == [
        True,
        True,
        False,
    ]


def test_polygon_on_boundary(polygon):
    assert polygon.on_boundary(np.array([(0, 0), (5, 0), (0, 5), (5, 5)])).tolist() == [
        1,
        1,
        1,
        1,
    ]
    assert polygon.on_boundary(np.array([(1, 1), (3, 3)])).tolist() == [0, 0]


def test_polygon_random_boundary_points(polygon):
    points = polygon.random_boundary_points(10)
    assert points.shape == (10, 2)


def test_polygon_is_rectangle(polygon):
    assert polygon.is_rectangle() == True  # Given this is a square.


@pytest.mark.parametrize(
    "input_coords, r_expected, theta_expected",
    [
        (
            np.array([[1, 1], [2, 2]]),
            (np.sqrt(2), 2 * np.sqrt(2)),
            (np.pi / 4, np.pi / 4),
        ),
        (np.array([[0, 1], [1, 0]]), (1, 1), (np.pi / 2, 0)),
        (np.array([[0, 0]]), (0, 0), (0, 0)),
    ],
)
def test_polar(input_coords, r_expected, theta_expected):
    r, theta = polar(input_coords)
    assert np.isclose(r, r_expected).all()
    assert np.isclose(theta, theta_expected).all()
