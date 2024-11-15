import pytest

from multipinn.geometry.domain import *


# Mock implementations of Domain for testing
class MockDomain(Domain):
    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype="float32")
        i = 0
        while i < n:
            tmp = (
                sample(n, self.dim, random) * (self.bbox[1] - self.bbox[0])
                + self.bbox[0]
            )
            tmp = tmp[self.inside(tmp)]

            if len(tmp) > n - i:
                tmp = np.random.permutation(tmp)
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def inside(self, x):
        return np.all(x >= self.bbox[0]) & np.all(x <= self.bbox[1])

    def on_boundary(self, x):
        return np.any(np.isclose(x, self.bbox[0])) | np.any(np.isclose(x, self.bbox[1]))

    def random_boundary_points(self, n, random="pseudo"):
        # Generate random points on boundaries as a simple example
        points = np.random.uniform(self.bbox[0], self.bbox[1], (n, self.dim))
        return points[np.random.choice(points.shape[0], n, replace=False), :]

    def boundary_normal(self, x):
        normal = np.zeros_like(x)
        if np.any(np.isclose(x, self.bbox[0])):
            normal[:, 0] = -1  # Facing negative x-boundary
        if np.any(np.isclose(x, self.bbox[1])):
            normal[:, 0] = 1  # Facing positive x-boundary
        return normal


@pytest.fixture
def create_mock_domain():
    return MockDomain(dim=2, bbox=(np.array([0, 0]), np.array([1, 1])), diam=1)


@pytest.fixture
def create_two_mock_domains():
    domain1 = MockDomain(dim=2, bbox=(np.array([0, 0]), np.array([1, 1])), diam=1)
    domain2 = MockDomain(
        dim=2, bbox=(np.array([0.5, 0.5]), np.array([1.5, 1.5])), diam=1
    )
    return domain1, domain2


def test_domain_union(create_two_mock_domains):
    domain1, domain2 = create_two_mock_domains
    union = DomainUnion(domain1, domain2)

    # Test inside method
    points_inside = np.array([[0.5, 0.5], [0.75, 0.75], [0.6, 0.6]])
    assert np.all(union.inside(points_inside))

    points_outside = np.array([[1.5, 1.5], [2, 2], [0.1, 0.1]])
    assert not np.any(union.inside(points_outside))


def test_domain_difference(create_two_mock_domains):
    domain1, domain2 = create_two_mock_domains
    difference = DomainDifference(domain1, domain2)

    points_inside = np.array([[0.1, 0.1], [0.9, 0.9]])
    assert np.all(difference.inside(points_inside))

    points_outside = np.array([[0.6, 0.6], [0.8, 0.8]])
    assert not np.any(difference.inside(points_outside))


def test_domain_intersection(create_two_mock_domains):
    domain1, domain2 = create_two_mock_domains
    intersection = DomainIntersection(domain1, domain2)

    points_inside = np.array([[0.6, 0.6], [0.75, 0.75]])
    assert np.all(intersection.inside(points_inside))

    points_outside = np.array([[0.4, 0.4], [1.2, 1.2]])
    assert not np.any(intersection.inside(points_outside))


def test_domain_product(create_two_mock_domains):
    domain1, domain2 = create_two_mock_domains
    product = DomainProduct(domain1, domain2)

    points_inside = np.array([[0.5, 0.5, 0.75, 0.75], [0.9, 0.9, 0.85, 1.5]])
    assert np.all(product.inside(points_inside))

    points_outside = np.array([[1.5, 1.5, 0.2, 0.3], [0.1, 0.1, 1.5, 1.5]])
    assert not np.any(product.inside(points_outside))


if __name__ == "__main__":
    pytest.main()
