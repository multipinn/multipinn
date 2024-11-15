import abc
from typing import Union

import numpy as np

from multipinn.generation.sampler import sample

from .geometry import Geometry


class Domain(Geometry):
    def __init__(self, dim, bbox, diam):
        super().__init__(dim, bbox, diam)
        self.idstr = type(self).__name__

    @abc.abstractmethod
    def inside(self, x) -> Union[np.ndarray, bool]:
        """Check if x is inside the geometry (including the boundary)."""

    @abc.abstractmethod
    def on_boundary(self, x) -> Union[np.ndarray, bool]:
        """Check if x is on the geometry boundary."""

    def strictly_inside(self, x) -> Union[np.ndarray, bool]:
        return np.logical_and(self.inside(x), ~self.on_boundary(x))

    def distance2boundary(self, x, dirn):
        raise NotImplementedError(
            "{}.distance2boundary to be implemented".format(self.idstr)
        )

    def mindist2boundary(self, x):
        raise NotImplementedError(
            "{}.mindist2boundary to be implemented".format(self.idstr)
        )

    def boundary_normal(self, x):
        """Compute the unit normal at x for Neumann or Robin boundary conditions."""
        raise NotImplementedError(
            "{}.boundary_normal to be implemented".format(self.idstr)
        )

    @abc.abstractmethod
    def random_boundary_points(self, n, random="pseudo"):
        """Compute the random point locations on the boundary."""
        raise NotImplementedError(
            "{}.random_boundary_points to be implemented".format(self.idstr)
        )

    def background_points(self, x, dirn, dist2npt, shift):
        raise NotImplementedError(
            "{}.background_points to be implemented".format(self.idstr)
        )

    def union(self, other: "Domain"):
        return DomainUnion(self, other)

    def __or__(self, other: "Domain"):
        return self.union(other)

    def difference(self, other: "Domain"):
        return DomainDifference(self, other)

    def __sub__(self, other: "Domain"):
        return self.difference(other)

    def intersection(self, other: "Domain"):
        return DomainIntersection(self, other)

    def __and__(self, other: "Domain"):
        return self.intersection(other)

    def product(self, other: "Domain"):
        return DomainProduct(self, other)

    def __mul__(self, other: "Domain"):
        return self.product(other)


class DomainUnion(Domain):
    def __init__(self, *geoms: Domain):
        if not geoms:
            raise ValueError("At least one domain is required")
        if not all(geom.dim == geoms[0].dim for geom in geoms):
            raise ValueError("All domains must have the same dimension")
        super().__init__(
            geoms[0].dim, self._compute_bbox(geoms), sum(geom.diam for geom in geoms)
        )
        self.geoms = []
        for geom in geoms:
            if isinstance(geom, DomainUnion):
                self.geoms += geom.geoms  # unpack recursive DomainUnion
            else:
                self.geoms.append(geom)

    @staticmethod
    def _compute_bbox(geoms):
        min_bbox = geoms[0].bbox[0]
        max_bbox = geoms[0].bbox[1]
        for geom in geoms[1:]:
            min_bbox = np.minimum(min_bbox, geom.bbox[0])
            max_bbox = np.maximum(max_bbox, geom.bbox[1])
        return min_bbox, max_bbox

    def inside(self, x):
        return np.any([geom.inside(x) for geom in self.geoms], axis=0)

    def on_boundary(self, x):
        # Be careful here! Boundaries "destroy" each other
        # This way [0 1] | [1 2] == [0 2] (without making boundary at [1 1])
        # However, [0 1] | [0 2] == (0 2] (without making boundary at [0 0]
        # In case abow, ues [0 1] | ([0 2] - [0 1]) instead
        # In general, in case of overlapping boundaries use A | (B-A) instead of A | B
        boundary_masks = [geom.on_boundary(x) for geom in self.geoms]
        inside_masks = [geom.inside(x) for geom in self.geoms]
        boundary_sum = np.sum(boundary_masks, axis=0)
        inside_sum = np.sum(inside_masks, axis=0)
        return (boundary_sum == 1) & (inside_sum == 1)

    def boundary_normal(self, x):
        # assert np.all(self.on_boundary(x))
        boundary_masks = [geom.on_boundary(x) for geom in self.geoms]
        boundary_normals = [geom.boundary_normal(x) for geom in self.geoms]
        result = np.zeros_like(boundary_normals[0])
        for mask, normal in zip(boundary_masks, boundary_normals):
            result += normal * mask[:, np.newaxis]
        return result

    def random_points(self, n, sampler="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype="float32")
        i = 0
        while i < n:
            tmp = (
                sample(n, self.dim, sampler) * (self.bbox[1] - self.bbox[0])
                + self.bbox[0]
            )
            tmp = tmp[self.inside(tmp)]

            if len(tmp) > n - i:
                tmp = np.random.permutation(tmp)
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def _split_n(self, n):
        weights = [
            (geom.diam / self.geoms[0].diam) ** (self.dim - 1)
            for geom in self.geoms[1:]
        ]
        k = n / (1 + sum(weights))
        n_values = [round(weight * k) for weight in weights]
        n_values.insert(0, n - sum(n_values))
        return n_values

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype="float32")
        n_values = self._split_n(n)
        i = 0
        while i < n:
            new_points = [
                geom.random_boundary_points(c, random=random)
                for geom, c in zip(self.geoms, n_values)
            ]
            tmp = np.concatenate(new_points, axis=0)
            tmp = tmp[self.on_boundary(tmp)]

            if len(tmp) > n - i:
                tmp = np.random.permutation(tmp)
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x


class DomainDifference(Domain):
    def __init__(self, geom1: Domain, geom2: Domain):
        if geom1.dim != geom2.dim:
            raise ValueError(
                "{} - {} failed (dimensions do not match).".format(
                    geom1.idstr, geom2.idstr
                )
            )
        super().__init__(geom1.dim, geom1.bbox, geom1.diam)
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x):
        return np.logical_or(
            np.logical_and(self.geom1.inside(x), ~self.geom2.inside(x)),
            np.logical_and(self.geom2.on_boundary(x), self.geom1.strictly_inside(x)),
        )

    def on_boundary(self, x):
        return np.logical_or(
            np.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x)),
            np.logical_and(self.geom2.on_boundary(x), self.geom1.strictly_inside(x)),
        )

    def boundary_normal(self, x):
        mask1 = np.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x))
        result = mask1[:, np.newaxis] * self.geom1.boundary_normal(x)
        mask2 = np.logical_and(self.geom2.on_boundary(x), self.geom1.strictly_inside(x))
        result += mask2[:, np.newaxis] * self.geom2.boundary_normal(x)
        return result

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype="float32")
        i = 0
        while i < n:
            tmp = self.geom1.random_points(n, random=random)
            ok = np.logical_or(
                ~self.geom2.inside(tmp),
                np.logical_and(
                    self.geom2.on_boundary(tmp), ~self.geom1.on_boundary(tmp)
                ),
            )  # Simular to self.inside(tmp), but we don't check geom1.inside(tmp)
            tmp = tmp[ok]

            if len(tmp) > n - i:
                tmp = np.random.permutation(tmp)
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype="float32")
        n1 = round(n * 1 / (1 + (self.geom2.diam / self.geom1.diam) ** (self.dim - 1)))
        n2 = n - n1
        i = 0
        while i < n:
            geom1_boundary_points = self.geom1.random_boundary_points(n1, random=random)
            geom1_boundary_points = geom1_boundary_points[
                ~self.geom2.inside(geom1_boundary_points)
            ]

            geom2_boundary_points = self.geom2.random_boundary_points(n2, random=random)
            geom2_boundary_points = geom2_boundary_points[
                self.geom1.strictly_inside(geom2_boundary_points)
            ]
            tmp = np.concatenate((geom1_boundary_points, geom2_boundary_points))

            if len(tmp) > n - i:
                tmp = np.random.permutation(tmp)
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x


class DomainIntersection(Domain):
    def __init__(self, geom1: Domain, geom2: Domain):
        if geom1.dim != geom2.dim:
            raise ValueError(
                "{} & {} failed (dimensions do not match).".format(
                    geom1.idstr, geom2.idstr
                )
            )
        super().__init__(
            geom1.dim,
            (
                np.maximum(geom1.bbox[0], geom2.bbox[0]),
                np.minimum(geom1.bbox[1], geom2.bbox[1]),
            ),
            min(geom1.diam, geom2.diam),
        )
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x):
        return np.logical_and(self.geom1.inside(x), self.geom2.inside(x))

    def on_boundary(self, x):
        return np.logical_or(
            np.logical_and(self.geom1.on_boundary(x), self.geom2.inside(x)),
            np.logical_and(self.geom2.on_boundary(x), self.geom1.inside(x)),
        )

    def boundary_normal(self, x):
        mask1 = np.logical_and(self.geom1.on_boundary(x), self.geom2.inside(x))
        result = mask1[:, np.newaxis] * self.geom1.boundary_normal(x)
        mask2 = np.logical_and(self.geom2.on_boundary(x), self.geom1.inside(x))
        result += mask2[:, np.newaxis] * self.geom2.boundary_normal(x)
        return result

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype="float32")
        i = 0
        while i < n:
            tmp = self.geom1.random_points(n, random=random)
            tmp = tmp[self.geom2.inside(tmp)]

            if len(tmp) > n - i:
                tmp = np.random.permutation(tmp)
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype="float32")
        n1 = round(n * 1 / (1 + (self.geom2.diam / self.geom1.diam) ** (self.dim - 1)))
        n2 = n - n1
        i = 0
        while i < n:
            geom1_boundary_points = self.geom1.random_boundary_points(n1, random=random)
            geom1_boundary_points = geom1_boundary_points[
                self.geom2.inside(geom1_boundary_points)
            ]

            geom2_boundary_points = self.geom2.random_boundary_points(n2, random=random)
            geom2_boundary_points = geom2_boundary_points[
                self.geom1.inside(geom2_boundary_points)
            ]
            tmp = np.concatenate((geom1_boundary_points, geom2_boundary_points))

            if len(tmp) > n - i:
                tmp = np.random.permutation(tmp)
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x


class DomainProduct(Domain):
    def __init__(self, geom1: Domain, geom2: Domain):
        super().__init__(
            geom1.dim + geom2.dim,
            (
                np.concatenate((geom1.bbox[0], geom2.bbox[0])),
                np.concatenate((geom1.bbox[1], geom2.bbox[1])),
            ),
            np.hypot(geom1.diam, geom2.diam),
        )
        self.geom1 = geom1
        self.geom2 = geom2

    def __split(self, x):
        return x[:, : self.geom1.dim], x[:, self.geom1.dim :]

    def inside(self, x):
        x1, x2 = self.__split(x)
        return np.logical_and(
            self.geom1.inside(x1),
            self.geom2.inside(x2),
        )

    def on_boundary(self, x) -> bool:
        x1, x2 = self.__split(x)
        return np.logical_or(
            self.geom1.on_boundary(x1),
            self.geom2.on_boundary(x2),
        )

    def boundary_normal(self, x):
        x1, x2 = self.__split(x)
        norm1 = self.geom1.on_boundary(x1)[:, np.newaxis] * self.geom1.boundary_normal(
            x1
        )
        norm2 = (
            self.geom2.on_boundary(x2)[:, np.newaxis] * self.geom2.boundary_normal(x2),
        )
        return np.concatenate((norm1, norm2), axis=1)

    def random_points(self, n, random="pseudo"):
        return np.concatenate(
            (self.geom1.random_points(n, random), self.geom2.random_points(n, random)),
            axis=1,
        )

    def random_boundary_points(self, n, random="pseudo"):
        # weight_1 = geom2.diam ** geom2.dim * geom1.diam ** (geom1.dim - 1)
        # weight_2 = geom1.diam ** geom1.dim * geom2.diam ** (geom2.dim - 1)
        n1 = round(n * 1 / (1 + self.geom1.diam / self.geom2.diam))
        n2 = n - n1
        result1 = np.concatenate(
            (
                self.geom1.random_boundary_points(n1, random),
                self.geom2.random_points(n1, random),
            ),
            axis=1,
        )
        result2 = np.concatenate(
            (
                self.geom1.random_points(n2, random),
                self.geom2.random_boundary_points(n2, random),
            ),
            axis=1,
        )
        return np.concatenate((result1, result2), axis=0)
