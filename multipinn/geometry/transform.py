from typing import Tuple, Union

import numpy as np

from .domain import Domain


class DomainShift(Domain):
    """A domain transformation that shifts the original domain by a given vector.

    This class creates a new domain by translating all points of the original domain
    by a fixed shift vector.

    Args:
        domain (Domain): The original domain to be shifted.
        shift (Union[np.ndarray, Tuple]): The vector by which to shift the domain.
            Must have the same dimension as the domain.
    """

    def __init__(self, domain: Domain, shift: Union[np.ndarray, Tuple]):
        assert len(shift) == domain.dim
        self.domain = domain
        self.shift = np.array(shift, dtype="float32")
        low = domain.bbox[0] + self.shift
        high = domain.bbox[1] + self.shift
        self.shift = self.shift[np.newaxis, :]
        super().__init__(domain.dim, (low, high), domain.diam)

    def inside(self, x) -> Union[np.ndarray, bool]:
        return self.domain.inside(x - self.shift)

    def on_boundary(self, x) -> Union[np.ndarray, bool]:
        return self.domain.on_boundary(x - self.shift)

    def boundary_normal(self, x):
        return self.domain.boundary_normal(x - self.shift)

    def random_boundary_points(self, n, random="pseudo"):
        return self.domain.random_boundary_points(n, random) + self.shift

    def random_points(self, n, random="pseudo"):
        return self.domain.random_points(n, random) + self.shift


class DomainPermute(Domain):
    """A domain transformation that permutes the coordinates of the original domain.

    This class creates a new domain by reordering the coordinates of points in the
    original domain according to a specified permutation.

    Args:
        domain (Domain): The original domain to be permuted.
        permutation (Tuple): A tuple specifying the new order of coordinates.
            Must be a permutation of (0, 1, ..., dim-1).
    """

    def __init__(self, domain: Domain, permutation: Tuple):
        assert len(permutation) == domain.dim
        self.domain = domain
        self.permutation = permutation
        self._inverse_permutation = tuple(np.argsort(permutation))
        bbox = (domain.bbox[0].take(permutation), domain.bbox[1].take(permutation))
        super().__init__(domain.dim, bbox, domain.diam)

    def forward(self, x):
        return x[:, self.permutation]

    def backward(self, x):
        return x[:, self._inverse_permutation]

    def inside(self, x) -> Union[np.ndarray, bool]:
        return self.domain.inside(self.backward(x))

    def on_boundary(self, x) -> Union[np.ndarray, bool]:
        return self.domain.on_boundary(self.backward(x))

    def boundary_normal(self, x):
        return self.forward(self.domain.boundary_normal(self.backward(x)))

    def random_boundary_points(self, n, random="pseudo"):
        return self.forward(self.domain.random_boundary_points(n, random))

    def random_points(self, n, random="pseudo"):
        return self.forward(self.domain.random_points(n, random))


class DomainAxisymmetricExtension(Domain):
    """A domain transformation that extends a domain by axial symmetry.

    This class creates a new domain by extending the original domain into one higher
    dimension through axial symmetry. The transformation creates parallel 2D planes
    where points rotate around all axes except the specified axis.

    Note:
        The specified axis is NOT the axis of rotation. Instead, rotation occurs
        around all axes EXCEPT the specified axis.

    Args:
        domain (Domain): The original domain to be extended.
        axis (int): The non-rotation axis index. Must be between 0 and domain.dim-1.
    """

    def __init__(self, domain, axis):
        assert 0 <= axis < domain.dim
        self.domain = domain
        self.axis = axis
        plane_low = min(domain.bbox[0][axis], -domain.bbox[1][axis])
        plane_high = max(domain.bbox[1][axis], -domain.bbox[0][axis])
        new_bbox = (
            np.concatenate(
                (
                    domain.bbox[0][:axis],
                    [plane_low],
                    domain.bbox[0][axis + 1 :],
                    [plane_low],
                )
            ),
            np.concatenate(
                (
                    domain.bbox[1][:axis],
                    [plane_high],
                    domain.bbox[1][axis + 1 :],
                    [plane_high],
                )
            ),
        )
        rest_diam2 = (
            domain.diam**2 - (domain.bbox[1][axis] - domain.bbox[0][axis]) ** 2
        )
        new_diam = np.sqrt(rest_diam2 + (new_bbox[1][axis] - new_bbox[0][axis]) ** 2)
        super().__init__(domain.dim + 1, new_bbox, new_diam)

    def projection(self, x):
        d = np.hypot(x[:, self.axis], x[:, -1])
        return np.concatenate(
            (x[:, : self.axis], d[:, np.newaxis], x[:, self.axis + 1 : -1]), axis=1
        )

    def extend(self, x, alpha):
        return np.concatenate(
            (
                x[:, : self.axis],
                x[:, self.axis : self.axis + 1] * np.cos(alpha)[:, np.newaxis],
                x[:, self.axis + 1 :],
                x[:, self.axis : self.axis + 1] * np.sin(alpha)[:, np.newaxis],
            ),
            axis=1,
        )

    def random_extend(self, x):
        alpha = np.random.uniform(0, 2 * np.pi, size=x.shape[0]).astype("float32")
        return self.extend(x, alpha)

    def _prob(self, x):
        result = np.abs(x[:, self.axis])
        return result / np.sum(result)

    def inside(self, x) -> Union[np.ndarray, bool]:
        return self.domain.inside(self.projection(x))

    def on_boundary(self, x) -> Union[np.ndarray, bool]:
        return self.domain.on_boundary(self.projection(x))

    def boundary_normal(self, x):
        alpha = np.arctan2(x[:, -1], x[:, self.axis])
        return self.extend(self.domain.boundary_normal(self.projection(x)), alpha)

    def random_boundary_points(self, n, random="pseudo"):
        tmp = self.domain.random_boundary_points(2 * n, random)
        index = np.random.choice(tmp.shape[0], n, p=self._prob(tmp))
        return self.random_extend(tmp[index])

    def random_points(self, n, random="pseudo"):
        tmp = self.domain.random_points(2 * n, random)
        index = np.random.choice(tmp.shape[0], n, p=self._prob(tmp))
        return self.random_extend(tmp[index])
