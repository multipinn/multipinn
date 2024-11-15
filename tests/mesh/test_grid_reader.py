import os
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import pytest

from multipinn.mesh.grid_reader import *


@pytest.fixture
def cleanup_files():
    files_to_cleanup = []

    yield files_to_cleanup

    for file_path in files_to_cleanup:
        if os.path.exists(file_path):
            os.remove(file_path)


@pytest.fixture
def sample_face():
    connection = Connections(
        indexes=[0, 1, 2],
        cells=[0, 1],
        normal=np.array([1, 0]),
        middle_point=np.array([0.5, 0.5]),
    )
    return Face(
        zone_id=1,
        first_index=0,
        last_index=2,
        type=FaceType.WALL,
        element_type=FaceElementType.LINEAR,
        connections=[connection],
        points=[np.array([0, 0]), np.array([1, 0]), np.array([0, 1])],
    )


@pytest.fixture
def sample_grid(sample_face):
    # This fixture creates a simple grid for testing
    return Grid(points=[[0, 0], [1, 0], [0, 1]], faces=[sample_face], dim=2)


@pytest.fixture
def mesh_data():
    return """
(0 "m41m41")
(2 3)
(10 (0 1 1 0))
(13 (0 1 1 0))
(12 (0 1 4 0))
(10 (5 1 1 2 3) (
-0.5 -0.5 -0.5
-0.5 -0.5 -0.5
))
(13 (1 1 1 3 3) (
1 1 1 1 0
1 1 1 1 0
))
(12 (3 1 4 11 2))
"""


def test_face_initialization(sample_face):
    assert sample_face.zone_id == 1
    assert sample_face.type == FaceType.WALL
    assert len(sample_face.connections) == 1
    assert sample_face.connections[0].indexes == [0, 1, 2]


def test_connection_normal(sample_face):
    conn = sample_face.connections[0]
    assert np.array_equal(conn.normal, np.array([1, 0]))


def test_grid_initialization(sample_grid):
    assert len(sample_grid.points) == 3
    assert len(sample_grid.faces) == 1
    assert sample_grid.dim == 2


def test_get_face_by_id(sample_grid):
    face = sample_grid.get_face_by_id(1)
    assert face is not None
    assert face.zone_id == 1


def test_grid_reader(mesh_data):
    with patch("builtins.open", mock_open(read_data=mesh_data)):
        reader = GridReader()
        grid = reader.read("dummy_path")
        assert grid.dim == 3
        assert len(grid.faces) > 0


def test_grid_msh_to_pt(mesh_data, cleanup_files, tmp_path):
    dummy_path = tmp_path / "dummy_path"
    pt_file_path = dummy_path.with_suffix(".pt")

    cleanup_files.append(pt_file_path)

    with patch("builtins.open", mock_open(read_data=mesh_data)):
        grid_msh_to_pt(str(dummy_path))
        assert pt_file_path.exists()
