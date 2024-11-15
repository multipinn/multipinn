import abc

import numpy as np


def isclose(a, b):
    """Check if two values are approximately equal within a small tolerance.

    Args:
        a: First value to compare
        b: Second value to compare

    Returns:
        bool: True if values are approximately equal within tolerance
    """
    atol = 1e-6
    return np.isclose(a, b, atol=atol)


class Geometry(abc.ABC):
    """Abstract base class for geometric objects.

    Provides core functionality for handling geometric shapes and domains.

    Args:
        dim (int): Dimension of the geometry
        bbox (tuple): Bounding box as ((min_x, min_y, ...), (max_x, max_y, ...))
        diam (float): Characteristic diameter/size of the geometry
    """

    def __init__(self, dim, bbox, diam):
        self.dim = dim
        self.bbox = bbox
        self.diam = min(diam, np.linalg.norm(bbox[1] - bbox[0]))

    @abc.abstractmethod
    def random_points(self, n, random="pseudo"):
        """Generate random points within or on the geometry.

        Args:
            n (int): Number of points to generate
            random (str, optional): Random number generation method. Options are:
                - "pseudo": Pseudorandom sampling (default)
                - "LHS": Latin Hypercube Sampling
                - "Halton": Halton sequence
                - "Hammersley": Hammersley sequence
                - "Sobol": Sobol sequence

        Returns:
            np.ndarray: Array of random points with shape (n, dim)

        Raises:
            NotImplementedError: Must be implemented by subclasses
            ValueError: If specified sampling method is not supported
        """
        raise NotImplementedError
