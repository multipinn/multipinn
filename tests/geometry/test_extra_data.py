from unittest.mock import MagicMock

import pytest
import torch

from multipinn.geometry.extra_data import ExtraData
from multipinn.geometry.shell import BaseShell


# Mock classes for testing
class MockBaseShell(BaseShell):
    def boundary_normal(self, points):
        return points + 1  # Simple mock transformation


class MockGeometry:
    pass  # No specific behavior needed for this test


@pytest.fixture
def mock_shell():
    shell = MagicMock(MockBaseShell)
    shell.boundary_normal.return_value = [4.0, 5.0, 6.0]
    return shell


@pytest.fixture
def extra_data_instance(mock_shell):
    return ExtraData(geometry=mock_shell, keys=["normals"])


def test_check_keys_valid(extra_data_instance):
    assert extra_data_instance.keys == ["normals"]


def test_check_keys_invalid_key(mock_shell):
    with pytest.raises(AssertionError):  # Should raise assertion error for invalid key
        ExtraData(geometry=mock_shell, keys=["invalid_key"])


def test_check_keys_no_base_shell():
    with pytest.raises(
        AssertionError
    ):  # Should raise assertion error if 'normals' used without BaseShell
        ExtraData(geometry=MockGeometry(), keys=["normals"])


def test_call_with_normals(extra_data_instance):
    points = torch.tensor([1.0, 2.0, 3.0])  # Sample input points
    result = extra_data_instance(points)

    expected_result = torch.tensor([4.0, 5.0, 6.0])
    assert len(result) == 1
    torch.testing.assert_close(result[0], expected_result)


def test_call_with_no_keys(mock_shell):
    extra_data = ExtraData(geometry=mock_shell, keys=[])
    points = torch.tensor([1.0, 2.0, 3.0])
    result = extra_data(points)
    assert result == ()
