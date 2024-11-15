import numpy as np
import torch

from multipinn.generation.sampler import sample

from .domain import Domain
from .geometry import isclose


class Ellipse(Domain):
    """Ellipse.

    Args:
        center: Center of the ellipse.
        semimajor: Semimajor of the ellipse.
        semiminor: Semiminor of the ellipse.
        angle: Rotation angle of the ellipse. A positive angle rotates the ellipse
            clockwise about the center and a negative angle rotates the ellipse
            counterclockwise about the center.
    """

    def __init__(self, center, semimajor, semiminor, angle=0):
        self.center = np.array(center, dtype="float32")
        self.semimajor = semimajor
        self.semiminor = semiminor
        self.angle = angle
        self.c = (semimajor**2 - semiminor**2) ** 0.5

        self.focus1 = np.array(
            [
                center[0] - self.c * np.cos(angle),
                center[1] + self.c * np.sin(angle),
            ],
            dtype="float32",
        )
        self.focus2 = np.array(
            [
                center[0] + self.c * np.cos(angle),
                center[1] - self.c * np.sin(angle),
            ],
            dtype="float32",
        )
        self.rotation_mat = np.array(
            [[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]
        )
        (
            self.theta_from_arc_length,
            self.total_arc,
        ) = self._theta_from_arc_length_constructor()
        super().__init__(
            2, (self.center - semimajor, self.center + semiminor), 2 * self.c
        )

    def on_boundary(self, x):
        d1 = np.linalg.norm(x - self.focus1, axis=-1)
        d2 = np.linalg.norm(x - self.focus2, axis=-1)
        return isclose(d1 + d2, 2 * self.semimajor)

    def inside(self, x):
        d1 = np.linalg.norm(x - self.focus1, axis=-1)
        d2 = np.linalg.norm(x - self.focus2, axis=-1)
        d = d1 + d2 - 2 * self.semimajor
        return np.logical_or(d <= 0, isclose(d, 0))

    def _ellipse_arc(self):
        """Cumulative arc length of ellipse with given dimensions. Returns theta values,
        distance cumulated at each theta, and total arc length.
        """
        # Divide the interval [0 , theta] into n steps at regular angles
        theta = np.linspace(0, 2 * np.pi, 10000)
        coords = np.array(
            [self.semimajor * np.cos(theta), self.semiminor * np.sin(theta)]
        )
        # Compute vector distance between each successive point
        coords_diffs = np.diff(coords)
        # Compute the full arc
        delta_r = np.linalg.norm(coords_diffs, axis=0)
        cumulative_distance = np.concatenate(([0], np.cumsum(delta_r)))
        c = np.sum(delta_r)
        return theta, cumulative_distance, c

    def _theta_from_arc_length_constructor(self):
        """Constructs a function that returns the angle associated with a given
        cumulative arc length for given ellipse.
        """
        theta, cumulative_distance, total_arc = self._ellipse_arc()

        # Construct the inverse arc length function
        def f(s):
            return np.interp(s, cumulative_distance, theta)

        return f, total_arc

    def random_points(self, n, random="pseudo"):
        # http://mathworld.wolfram.com/DiskPointPicking.html
        rng = sample(n, 2, random)
        r, theta = rng[:, 0], 2 * np.pi * rng[:, 1]
        x, y = self.semimajor * np.cos(theta), self.semiminor * np.sin(theta)
        X = np.sqrt(r) * np.vstack((x, y))
        # print(X.shape, 'inner_pnts')
        return np.matmul(self.rotation_mat, X).T + self.center

    def uniform_boundary_points(self, n):  # we use random_boundary_points instead
        # https://codereview.stackexchange.com/questions/243590/generate-random-points-on-perimeter-of-ellipse
        u = np.linspace(0, 1, num=n, endpoint=False).reshape((-1, 1))
        theta = self.theta_from_arc_length(u * self.total_arc)
        X = np.hstack((self.semimajor * np.cos(theta), self.semiminor * np.sin(theta)))
        return np.matmul(self.rotation_mat, X.T).T + self.center

    def random_boundary_points(self, n, random="pseudo"):
        u = sample(n, 1, random)
        theta = self.theta_from_arc_length(u * self.total_arc)
        X = np.hstack((self.semimajor * np.cos(theta), self.semiminor * np.sin(theta)))
        # print(X.shape, 'bound_pnts')
        # print(self.rotation_mat.shape, 'bound_pnts')
        return np.matmul(self.rotation_mat, X.T).T + self.center

    def boundary_normal(self, x):
        pnts = x - self.center
        pnts = (self.rotation_mat.T @ pnts.T).T
        grad = pnts / np.array(
            [self.semimajor**2, self.semiminor**2]
        )  # /np.linalg.norm()
        grad = (self.rotation_mat @ grad.T).T
        l = np.linalg.norm(grad, axis=1)
        grad = (grad.T / l).T
        return grad

    @staticmethod
    def is_valid(vertices):
        """Check if the geometry is a Rectangle."""
        return (
            len(vertices) == 4
            and isclose(np.prod(vertices[1] - vertices[0]), 0)
            and isclose(np.prod(vertices[2] - vertices[1]), 0)
            and isclose(np.prod(vertices[3] - vertices[2]), 0)
            and isclose(np.prod(vertices[0] - vertices[3]), 0)
        )


class Polygon(Domain):
    """Simple polygon.

    Args:
        vertices: The order of vertices can be in a clockwise or counterclockwise
            direction. The vertices will be re-ordered in counterclockwise (right hand
            rule).
    """

    def __init__(self, vertices):
        self.vertices = np.array(vertices, dtype="float32")
        self.nvertices = len(self.vertices)
        self.area = self.polygon_signed_area(self.vertices)
        if self.area < 0:  # make clockwise
            self.area = -self.area
            self.vertices = np.flipud(self.vertices)

        self.diagonals = np.linalg.norm(
            self.vertices[:, np.newaxis, :] - self.vertices, axis=2
        )
        super().__init__(
            2,
            (np.amin(self.vertices, axis=0), np.amax(self.vertices, axis=0)),
            np.max(self.diagonals),
        )
        self.perimeter = np.sum(
            self.diagonals[np.arange(self.nvertices) - 1, np.arange(self.nvertices)]
        )
        self.segments = self.vertices - np.roll(self.vertices, 1, axis=0)
        self.normal = np.array([self.segments[:, 1], -self.segments[:, 0]]).T
        self.normal = self.normal / np.linalg.norm(self.normal, axis=1).reshape(-1, 1)
        self._sempler_mult = 1.1 * np.prod(self.bbox[1] - self.bbox[0]) / self.area

    def inside(self, x):
        # https://github.com/sasamil/PointInPolygon_Py/blob/master/pointInside.py
        intersections = np.zeros(shape=len(x), dtype="float32")
        dx2 = x - self.vertices[np.newaxis, -1, :]
        for i in range(0, self.nvertices):
            dx = dx2
            dx2 = x - self.vertices[np.newaxis, i, :]
            cross = np.cross(dx, dx2, axis=-1)

            index = np.logical_xor(dx[:, 1] < 0, dx2[:, 1] < 0)
            intersections[index] += np.sign(cross[index])

            index = np.logical_and(isclose(cross, 0), np.all(dx * dx2 <= 0, axis=1))
            intersections[
                index
            ] = self.nvertices  # point on the border - it must be inside
        return intersections != 0

    def on_boundary(self, x):
        _on = np.zeros(shape=len(x), dtype=int)
        l2 = np.linalg.norm(self.vertices[-1] - x, axis=-1)
        for i in range(self.nvertices):
            l1 = l2
            l2 = np.linalg.norm(self.vertices[i] - x, axis=-1)
            _on[isclose(l1 + l2, self.diagonals[i - 1, i])] += 1
        return _on > 0

    def boundary_normal(self, x):
        result = np.zeros_like(x)
        l2 = np.linalg.norm(self.vertices[-1] - x, axis=-1)
        close_to_point = np.zeros(shape=len(x), dtype=bool)
        for i in range(self.nvertices):
            l1 = l2
            l2 = np.linalg.norm(self.vertices[i] - x, axis=-1)
            close_to_line = isclose(l1 + l2, self.diagonals[i - 1, i])
            result[close_to_line] += self.normal[i]
            close_to_point = np.logical_or(
                close_to_point, isclose(x, self.vertices[i]).all(axis=1)
            )
        result[close_to_point] = 0
        return result

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype="float32")
        size = self.bbox[1] - self.bbox[0]
        i = 0
        while i < n:
            tmp = (
                sample(round(n * self._sempler_mult), 2, sampler="pseudo") * size
                + self.bbox[0]
            )
            tmp = tmp[self.inside(tmp)]
            if len(tmp) > n - i:
                tmp = np.random.permutation(tmp)
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def uniform_boundary_points(self, n):  # we use random_boundary_points instead
        density = n / self.perimeter
        x = []
        for i in range(-1, self.nvertices - 1):
            x.append(
                np.linspace(
                    0,
                    1,
                    num=int(np.ceil(density * self.diagonals[i, i + 1])),
                    endpoint=False,
                )[:, None]
                * (self.vertices[i + 1] - self.vertices[i])
                + self.vertices[i]
            )
        x = np.vstack(x)
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_boundary_points(self, n, random="pseudo"):
        result = np.empty(shape=(n, self.dim), dtype="float32")
        r = sample(n, 2, random)
        selector, portion = r[:, 0], r[:, 1]
        selector *= self.perimeter
        s = 0.0
        for i in range(self.nvertices):
            new_s = s + self.diagonals[i - 1, i]
            index = np.logical_and(s <= selector, selector < new_s)
            s = new_s
            result[index] = (
                self.vertices[i - 1]
                + self.segments[np.newaxis, i, :] * portion[index, np.newaxis]
            )
        return result

    def is_rectangle(self):
        """Check if the geometry is a rectangle.

        https://stackoverflow.com/questions/2303278/find-if-4-points-on-a-plane-form-a-rectangle/2304031

        1. Find the center of mass of corner points: cx=(x1+x2+x3+x4)/4, cy=(y1+y2+y3+y4)/4
        2. Test if square of distances from center of mass to all 4 corners are equal
        """
        if self.nvertices != 4:
            return False

        c = np.mean(self.vertices, axis=0)
        d = np.sum((self.vertices - c) ** 2, axis=1)
        return np.allclose(d, np.full(4, d[0]))

    @staticmethod
    def polygon_signed_area(vertices):
        """The (signed) area of a simple polygon.

        If the vertices are in the counterclockwise direction, then the area is positive; if
        they are in the clockwise direction, the area is negative.

        Shoelace formula: https://en.wikipedia.org/wiki/Shoelace_formula
        """
        x, y = zip(*vertices)
        x = np.array(list(x) + [x[0]])
        y = np.array(list(y) + [y[0]])
        return 0.5 * (np.sum(x[:-1] * y[1:]) - np.sum(x[1:] * y[:-1]))


def polar(x):
    """Get the polar coordinated for a 2d vector in cartesian coordinates."""
    r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    theta = np.arctan2(x[:, 1], x[:, 0])
    return r, theta
