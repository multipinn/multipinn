from unittest.mock import mock_open, patch

import numpy as np
import pytest

from multipinn.mesh.comsol_reader import *

# Parameters and mock data
stationary_header = "This is a stationary header\n"
time_dependent_header = "u (K) @ t=0.0 u (K) @ t=0.1 u (K) @ t=0.2\n"
stationary_data = "0.0 1.0 5.0\n1.0 2.0 6.0\n"
time_dependent_data = "0.0 1.0 5.0\n1.0 2.0 6.0\n"

stationary_data_array = np.array([[0.0, 1.0, 5.0], [1.0, 2.0, 6.0]], dtype="float32")


# Parametrized tests for read_comsol_file
@pytest.mark.parametrize(
    "mock_header, is_stationary, expected_times, expected_points, expected_u_values",
    [
        (
            stationary_header + stationary_data,
            True,
            None,
            np.array([[0.0, 1.0], [1.0, 2.0]]),
            np.array([[5.0], [6.0]]),
        ),
        (
            time_dependent_header + time_dependent_data,
            False,
            np.array([0.0, 0.1, 0.2]),
            [
                np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0]]),
                np.array([[0.0, 1.0, 0.1], [1.0, 2.0, 0.1]]),
                np.array([[0.0, 1.0, 0.2], [1.0, 2.0, 0.2]]),
            ],
            np.array([[5.0], [6.0]]),
        ),
    ],
)
def test_read_comsol_file(
    mock_header, is_stationary, expected_times, expected_points, expected_u_values
):
    with patch("builtins.open", mock_open(read_data=mock_header)):
        with patch("numpy.genfromtxt", return_value=stationary_data_array):
            filepath = "dummy_path"
            t_values, points, u_values = read_comsol_file(
                filepath, is_stationary=is_stationary
            )

            if is_stationary:
                assert t_values is expected_times
                assert np.array_equal(points, expected_points)
                assert np.array_equal(u_values, expected_u_values)
            else:
                assert np.array_equal(t_values, expected_times)
                for p1, p2 in zip(points, expected_points):
                    assert np.array_equal(p1, p2)
                assert np.array_equal(u_values, expected_u_values)


# Parametrized tests for get_problem_type
@pytest.mark.parametrize(
    "header, expected_result",
    [
        (stationary_header, "stationary"),
        (time_dependent_header, "time-dependent"),
    ],
)
def test_get_problem_type(header, expected_result):
    with patch("builtins.open", mock_open(read_data=header)):
        assert get_problem_type("dummy_path") == expected_result
