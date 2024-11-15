import itertools

import numpy as np
from scipy import stats

from multipinn.generation.sampler import sample

from .domain import Domain
from .geometry import isclose


class Hypercube(Domain):
    def __init__(self, low, high):
        if len(low) != len(high):
            raise ValueError("Dimensions of low and high do not match.")

        self.low = np.array(low, dtype="float32")
        self.high = np.array(high, dtype="float32")
        if np.any(self.low > self.high):
            raise ValueError("low > high")

        self.side_length = self.high - self.low
        super().__init__(
            len(low), (self.low, self.high), np.linalg.norm(self.side_length)
        )
        self.volume = np.prod(self.side_length)
        if self.volume == 0:
            zero_indexes = self.side_length == 0  # if some side_lengths are 0
            self.areas = np.zeros_like(self.side_length)  # then all areas are 0
            if np.sum(zero_indexes) == 1:  # except if only one side_length is 0
                self.areas[zero_indexes] = np.prod(
                    self.side_length[~zero_indexes]
                )  # then surface area is not 0
            self.normalized_areas = np.zeros_like(
                self.side_length
            )  # sides with side_length 0 dominate other areas
            self.normalized_areas[zero_indexes] = 1 / np.sum(
                zero_indexes
            )  # but all dominating areas are equal
        else:
            self.areas = self.volume / self.side_length
            self.normalized_areas = self.areas / np.sum(self.areas)

    def inside(self, x):
        return np.logical_and(
            np.all(x >= self.low, axis=-1), np.all(x <= self.high, axis=-1)
        )

    def on_boundary(self, x):
        _on_boundary = np.logical_or(
            np.any(isclose(x, self.low), axis=-1),
            np.any(isclose(x, self.high), axis=-1),
        )
        return np.logical_and(self.inside(x), _on_boundary)

    def closest_point(self, x):
        return np.clip(x, self.low, self.high)

    def closest_boundary_point(self, x):
        unit_cube = (x - self.low) / (self.high - self.low)
        candidates = np.repeat(x[:, np.newaxis, :], self.dim, axis=1)
        for i in range(self.dim):
            candidates[:, i, i] = np.where(
                unit_cube[:, i] >= 0.5, self.high[i], self.low[i]
            )
        candidates = np.clip(
            candidates,
            self.low[np.newaxis, np.newaxis, :],
            self.high[np.newaxis, np.newaxis, :],
        )
        distances = np.linalg.norm(x[:, np.newaxis, :] - candidates, axis=2)
        closest_indices = np.argmin(distances, axis=1)
        closest_points = candidates[np.arange(x.shape[0]), closest_indices]
        return closest_points

    def boundary_normal(self, x):
        _n = -isclose(x, self.low).astype("float32") + isclose(x, self.high)
        # For vertices, the normal is averaged for all directions
        idx = np.count_nonzero(_n, axis=-1) > 1
        if np.any(idx):
            print(
                f"Warning: {self.__class__.__name__} boundary_normal called on vertices. "
                "You may use PDE(..., exclusions=...) to exclude the vertices."
            )
            l = np.linalg.norm(_n[idx], axis=-1, keepdims=True)
            _n[idx] /= l
        return _n

    def uniform_points(self, n, boundary=True):  # we use random_points instead
        # Are we doing the simular thing in Grid.smart_volume?
        dx = (self.volume / n) ** (1 / self.dim)
        xi = []
        for i in range(self.dim):
            ni = int(np.ceil(self.side_length[i] / dx))
            if boundary:
                xi.append(
                    np.linspace(self.low[i], self.high[i], num=ni, dtype="float32")
                )
            else:
                xi.append(
                    np.linspace(
                        self.low[i],
                        self.high[i],
                        num=ni + 1,
                        endpoint=False,
                        dtype="float32",
                    )[1:]
                )
        x = np.array(list(itertools.product(*xi)))
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_points(self, n, random="pseudo"):
        x = sample(n, self.dim, random)
        return self.side_length * x + self.low

    def random_boundary_points(self, n, random="pseudo"):
        x = sample(n, self.dim, random)
        # Randomly pick a dimension
        rand_dim = np.random.choice(self.dim, n, p=self.normalized_areas)
        # Replace value of the randomly picked dimension with the nearest boundary value (0 or 1)
        x[np.arange(n), rand_dim] = np.round(x[np.arange(n), rand_dim])
        return self.side_length * x + self.low


class Hypersphere(Domain):
    def __init__(self, center, radius):
        self.center = np.array(center, dtype="float32")
        self.radius = radius
        super().__init__(
            len(center), (self.center - radius, self.center + radius), 2 * radius
        )

        self._r2 = radius**2

    def inside(self, x):
        r = np.linalg.norm(x - self.center, axis=-1)
        return np.logical_or(r <= self.radius, isclose(r, self.radius))

    def on_boundary(self, x):
        return isclose(np.linalg.norm(x - self.center, axis=-1), self.radius)

    def closest_point(self, x):
        direction = x - self.center
        distance = np.linalg.norm(direction, axis=1)
        mask = distance > self.radius
        closest_points = x.copy()
        direction_normalized = direction[mask] / distance[mask, np.newaxis]
        closest_points[mask] = self.center + direction_normalized * self.radius
        return closest_points

    def closest_boundary_point(self, x):
        direction = x - self.center
        distance = np.linalg.norm(direction, axis=1)
        direction_normalized = direction / distance[:, np.newaxis]
        closest_boundary_points = self.center + direction_normalized * self.radius
        return closest_boundary_points

    def distance2boundary_unitdirn(self, x, dirn):
        # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        xc = x - self.center
        ad = np.dot(xc, dirn)
        return (-ad + (ad**2 - np.sum(xc * xc, axis=-1) + self._r2) ** 0.5).astype(
            "float32"
        )

    def distance2boundary(self, x, dirn):
        return self.distance2boundary_unitdirn(x, dirn / np.linalg.norm(dirn))

    def mindist2boundary(self, x):
        return np.amin(self.radius - np.linalg.norm(x - self.center, axis=1))

    def boundary_normal(self, x):
        return (x - self.center) / self.radius

    def random_points(self, n, random="pseudo"):
        # https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
        if random == "pseudo":
            U = np.random.rand(n, 1).astype("float32")
            X = np.random.normal(size=(n, self.dim)).astype("float32")
            # U = torch.rand((n, 1), dtype=torch.float32)
            # X = torch.randn(size=(n, self.dim), dtype=torch.float32)
        else:
            rng = sample(n, self.dim + 1, random)
            U, X = rng[:, 0:1], rng[:, 1:]  # Error if X = [0, 0, ...]
            X = stats.norm.ppf(X).astype("float32")
        X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
        X = U ** (1 / self.dim) * X
        return self.radius * X + self.center

    def random_boundary_points(self, n, random="pseudo"):
        # http://mathworld.wolfram.com/HyperspherePointPicking.html
        if random == "pseudo":
            X = np.random.normal(size=(n, self.dim)).astype("float32")
        else:
            U = sample(n, self.dim, random)  # Error for [0, 0, ...] or [0.5, 0.5, ...]
            X = stats.norm.ppf(U).astype("float32")
        X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
        return self.radius * X + self.center

    def background_points(self, x, dirn, dist2npt, shift):
        dirn = dirn / np.linalg.norm(dirn)
        dx = self.distance2boundary_unitdirn(x, -dirn)
        n = max(dist2npt(dx), 1)
        h = dx / n
        pts = x - np.arange(-shift, n - shift + 1, dtype="float32")[:, None] * h * dirn
        return pts
