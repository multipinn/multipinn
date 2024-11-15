from typing import Sequence

import torch

from .geometry import Geometry
from .shell import BaseShell


class ExtraData:
    """A class for computing additional geometric data like normal vectors for geometric shapes.

    This class provides functionality to calculate extra geometric properties based on the
    provided geometry and requested keys. Currently supports computation of normal vectors,
    with potential for extension to other geometric properties.

    Args:
        geometry (Geometry): The geometric shape to compute extra data for.
        keys (Sequence[str]): Sequence of strings specifying which extra data to compute.
            Currently supported keys:
                - "normals": Computes normal vectors (requires geometry to be a BaseShell)

    Raises:
        AssertionError: If invalid keys are provided or if geometry doesn't support
            the requested computations.
    """

    def __init__(self, geometry: Geometry, keys: Sequence[str]):
        self.geometry = geometry
        self.keys = keys
        self.check_keys()

    def check_keys(self):
        """Validates the requested keys and geometry compatibility.

        Checks that:
        1. All requested keys are in the list of supported keys
        2. For "normals" key: geometry is a BaseShell instance with boundary_normal method

        Raises:
            AssertionError: If any validation check fails
        """
        possible_keys = ["normals"]  # ['normals', 'edges']
        for key in self.keys:
            assert key in possible_keys
        if "normals" in self.keys:
            assert isinstance(self.geometry, BaseShell)
            assert hasattr(self.geometry, "boundary_normal")

    def __call__(self, points):
        """Computes the requested geometric properties for the given points.

        Args:
            points (torch.Tensor): Points at which to compute the geometric properties.
                Shape: (batch_size, dimension)

        Returns:
            tuple: Contains computed geometric properties in the order specified by keys.
                For "normals" key: Returns tensor of normal vectors at given points
                Shape: (batch_size, dimension)
        """
        result = []
        for key in self.keys:
            if key == "normals":
                result.append(
                    torch.tensor(
                        self.geometry.boundary_normal(points.detach().cpu().numpy())
                    )
                )
        return tuple(result)
