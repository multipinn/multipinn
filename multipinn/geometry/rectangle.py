from ..condition import Condition
# from ..generation.base_generator import BaseGenerator
from . import Geometry


class RectangleArea(Geometry):
    def __init__(self, low, high, generator = None):
        self.low = low
        self.high = high
        self.n_dims = len(self.low)
        self.generator = generator

    def generate_points(self, condition: Condition, model):
        return self.generator.generate(self, condition, model)
