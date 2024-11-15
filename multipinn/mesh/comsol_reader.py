import re
from typing import List, Tuple, Union

import numpy as np


def read_comsol_file(
    filepath: str, is_stationary: bool = True
) -> Tuple[Union[np.ndarray, None], List[np.ndarray], np.ndarray]:
    """Read solution data from a COMSOL export file.

    Parses a COMSOL-exported file containing mesh points and solution values. The file
    should contain coordinates (x, y) and solution values u(x, y) for stationary problems,
    or u(x, y, t) for time-dependent problems.

    Args:
        filepath (str): Path to the COMSOL export file.
        is_stationary (bool, optional): Whether the problem is stationary. Defaults to True.

    Returns:
        Tuple containing:
            - t_values (np.ndarray or None): Time values for time-dependent problems, None for stationary.
            - points (List[np.ndarray]): Mesh points. For stationary problems, contains a single array
              of (x, y) coordinates. For time-dependent problems, contains arrays of (x, y, t) coordinates
              for each time step.
            - u_values (np.ndarray): Solution values at each point.

    Raises:
        ValueError: If the file doesn't contain time values but is_stationary=False.
    """
    with open(filepath, "r") as f:
        header = f.readline()

    time_pattern = r"u \(K\) @ t=(\d+(?:\.\d+)?)"
    t_values = np.array([float(t) for t in re.findall(time_pattern, header)])

    data = np.genfromtxt(filepath, comments="%", dtype="float32")

    x, y = data[:, 0], data[:, 1]
    u_values = data[:, 2:]

    if is_stationary:
        points = np.column_stack((x, y))
        return None, points, u_values
    else:
        if len(t_values) == 0:
            raise ValueError(
                "File does not contain time values, but task is specified as non-stationary."
            )

        points = []
        for t in t_values:
            point = np.column_stack((x, y, np.full(x.shape, t)))
            points.append(point)

        return t_values, points, u_values


def get_problem_type(filepath: str) -> str:
    """Determine whether a COMSOL export file contains stationary or time-dependent data.

    Examines the header of the COMSOL file to check for time values and determines
    the problem type accordingly.

    Args:
        filepath (str): Path to the COMSOL export file.

    Returns:
        str: Either "stationary" or "time-dependent".
    """
    with open(filepath, "r") as f:
        header = f.readline()

    time_pattern = r"u \(K\) @ t=(\d+(?:\.\d+)?)"
    t_values = re.findall(time_pattern, header)

    return "stationary" if len(t_values) == 0 else "time-dependent"
