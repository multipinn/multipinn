import torch

from multipinn.condition.condition import Condition
from multipinn.generation.generator import Generator
from multipinn.mesh.mesh import MeshArea


class UniformGeneratorMesh(Generator):
    def __init__(self, n_points, method):
        self.n_points = n_points
        self.method = method

    def generate(self, geometry: MeshArea, condition: Condition, model):
        if self.method == "uniform_sample":
            return self.uniform_sample(geometry)
        else:
            raise Exception("unknown method = " + self.method)

    def uniform_sample(self, geometry: MeshArea):
        self.device = geometry.points.device
        indices = torch.randperm(geometry.points.shape[0])[: self.n_points].to(
            self.device
        )
        if "normals" in dir(geometry.points):
            return (
                geometry.points[indices].requires_grad_(),
                geometry.points.normals[indices].requires_grad_(),
            )
        else:
            return geometry.points[indices].requires_grad_(), None
