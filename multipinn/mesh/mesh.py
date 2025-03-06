import torch

from multipinn.condition.condition import Condition
from multipinn.generation.generator import Generator
from multipinn.geometry.geometry import Geometry
from multipinn.mesh.grid_reader import Face


class MeshStorage(Geometry):
    def __init__(self, points: torch.Tensor, generator: Generator = None):
        self.n_dims = points.shape[1]
        self.points = points
        self.generator = generator

    def generate_points(self, condition: Condition, model):
        return self.points, None


class MeshArea(Geometry):
    def __init__(self, face: Face, n_dims: int):
        self.n_connections = len(face.connections)
        self.n_dims = n_dims
        self.points = torch.zeros(self.n_connections, self.n_dims)

        if 0 in face.connections[0].cells:
            self.points.normals = torch.zeros(self.n_connections, self.n_dims)

        for i, connection in enumerate(face.connections):
            self.points[i] = torch.tensor(connection.middle_point) / 50
            if 0 in connection.cells:
                self.points.normals[i] = torch.tensor(connection.normal)

    def random_points(self, n, random="pseudo"):
        indices = torch.randperm(self.points.shape[0])[:n]
        sample_points = self.points[indices].requires_grad_()
        sample_points.normals = (
            self.points.normals[indices].requires_grad_()
            if "normals" in dir(self.points)
            else None
        )
        return sample_points
