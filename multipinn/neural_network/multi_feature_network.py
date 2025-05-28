from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from .activation_function import GELU


def _dot(a, b):
    return torch.sum(a * b, dim=1)


def polygon(points: List[torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns a function that computes the signed distance from 2D points to a polygon.
    The sign is positive inside the polygon and negative outside (using a simple winding test).
    """
    pts = [p.clone().float() for p in points]

    def f(p: torch.Tensor) -> torch.Tensor:
        n = len(pts)
        d2 = torch.sum((p - pts[0]) ** 2, dim=1)
        s = torch.ones(p.shape[0], dtype=p.dtype, device=p.device)

        for i in range(n):
            j = (i + n - 1) % n
            vi = pts[i]
            vj = pts[j]
            e = (vj - vi).reshape(1, 2)
            w = p - vi
            proj = _dot(w, e) / _dot(e, e)
            clamped = torch.clamp(proj, 0.0, 1.0).unsqueeze(-1)
            b = w - e * clamped
            d2 = torch.min(d2, torch.sum(b**2, dim=1))

            # winding sign test
            c1 = p[:, 1] >= vi[1]
            c2 = p[:, 1] < vj[1]
            c3 = e[0, 0] * w[:, 1] > e[0, 1] * w[:, 0]
            inside = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
            s = torch.where(inside, -s, s)

        return s * torch.sqrt(d2)

    return f


class ShiftV(nn.Module):
    def __init__(self, shift: float, scale: float = 1.0):
        super().__init__()
        self.shift = shift
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (x - self.shift)


class SquareShiftV(nn.Module):
    def __init__(self, shift: float, scale: float = 1.0):
        super().__init__()
        self.shift = shift
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (x - self.shift) ** 2


class SigmoidShiftV(nn.Module):
    def __init__(self, shift: float):
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x - self.shift)


class ScaledTrigShiftV(nn.Module):
    def __init__(self, trig_func: Callable, scale: float, shift: float = 0.0):
        super().__init__()
        self.trig_func = trig_func
        self.scale = scale
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trig_func(torch.pi * (x - self.shift) * self.scale)


class CyclicShift(nn.Module):
    """
    Cylindrical feature: squared distance in the plane of two coordinates.
    """

    def __init__(self, shift1: float, shift2: float, index1: int, index2: int):
        super().__init__()
        self.shift1 = shift1
        self.shift2 = shift2
        self.index1 = index1
        self.index2 = index2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, self.index1] - self.shift1) ** 2 + (
            x[:, self.index2] - self.shift2
        ) ** 2


class RBFShift(nn.Module):
    """
    1D radial basis function: exp(-width * (x[:,index] - shift)^2)
    """

    def __init__(self, shift: float, width: float, index: int):
        super().__init__()
        self.shift = shift
        self.width = width
        self.index = index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.width * (x[:, self.index] - self.shift) ** 2)


class RBF2D(nn.Module):
    """
    2D radial basis: exp(-width * ((x- cx)^2 + (y- cy)^2)) in the XY-plane.
    """

    def __init__(self, center: Tuple[float, float], width: float):
        super().__init__()
        self.center = torch.tensor(center, dtype=torch.float32)
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # project to XY-plane and compute squared distance
        d2 = (x[:, 0] - self.center[0]) ** 2 + (x[:, 1] - self.center[1]) ** 2
        return torch.exp(-self.width * d2)


class WallDistance(nn.Module):
    """
    Signed distance to 2D pipe boundary (extruded in Z).
    """

    def __init__(self, wall_fn: Callable):
        super().__init__()
        self.wall_fn = wall_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xy = x[:, :2]
        return self.wall_fn(xy)


class BoxBoundaryDistance(torch.nn.Module):
    """
    Для каждой точки x = (x,y,z) возвращает
      - если x внутри [mn, mx] по всем трём осям:
          min расстояние до любой из 6 плоскостей (то есть до любой из граней)
      - если снаружи: обычное euclid-расстояние до ближайшей точки на коробке
    """

    def __init__(self, min_corner: list, max_corner: list):
        super().__init__()
        mn = torch.tensor(min_corner, dtype=torch.float32)
        mx = torch.tensor(max_corner, dtype=torch.float32)
        self.register_buffer("mn", mn)
        self.register_buffer("mx", mx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = torch.max(self.mn - x, x - self.mx)  
        d_out = torch.norm(torch.clamp(o, min=0.0), dim=1)

        inside_mask = ((x >= self.mn) & (x <= self.mx)).all(dim=1)
        d1 = x - self.mn 
        d2 = self.mx - x  
        d_inside_axes = torch.min(d1, d2)  
        d_in = d_inside_axes.min(dim=1).values  

        return torch.where(inside_mask, d_in, d_out)


class UnionBoundaryDistance(torch.nn.Module):
    def __init__(self, boxes: list[tuple[list, list]]):
        """
        boxes: список [(min_corner, max_corner), ...]
        """
        super().__init__()
        self.boxes = torch.nn.ModuleList(
            [BoxBoundaryDistance(mn, mx) for mn, mx in boxes]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ds = torch.stack([b(x) for b in self.boxes], dim=1) 
        return ds.min(dim=1).values


class MultiFeatureEncoding(nn.Module):
    def __init__(self, functions: nn.ModuleList = None):
        super().__init__()
        self.vector_functions = nn.ModuleList(
            [
                ShiftV(4.5),
                SquareShiftV(4.5),
                SigmoidShiftV(4.5),
                ShiftV(1.5),
                SquareShiftV(1.5),
                SigmoidShiftV(1.5),
                nn.Identity(),
                SquareShiftV(0.0),
                SigmoidShiftV(0.0),
                ScaledTrigShiftV(torch.sin, 1 / 9, 4.5),
                ScaledTrigShiftV(torch.sin, 1 / 1, 0.5),
                ScaledTrigShiftV(torch.sin, 1 / 3, 1.5),
                ScaledTrigShiftV(torch.cos, 1 / 9, 4.5),
                ScaledTrigShiftV(torch.cos, 1 / 1, 0.5),
                ScaledTrigShiftV(torch.cos, 1 / 3, 1.5),
            ]
        )

        pipe = polygon(
            [
                torch.tensor([0, 0], dtype=torch.float32),
                torch.tensor([5, 0], dtype=torch.float32),
                torch.tensor([5, 2], dtype=torch.float32),
                torch.tensor([9, 2], dtype=torch.float32),
                torch.tensor([9, 3], dtype=torch.float32),
                torch.tensor([4, 3], dtype=torch.float32),
                torch.tensor([4, 1], dtype=torch.float32),
                torch.tensor([0, 1], dtype=torch.float32),
            ]
        )

        boxes = [
            ([0, 0, 0], [4, 1, 1]),
            ([4, 0, 0], [5, 3, 1]),
            ([5, 2, 0], [9, 3, 1]),
        ]

        self.functions = (
            functions
            if functions
            else nn.ModuleList(
                [
                    # cylindrical in YZ-plane (around two centers)
                    CyclicShift(0.5, 0.5, 1, 2),
                    CyclicShift(1.5, 0.5, 1, 2),
                    # cylindrical in XZ-plane
                    CyclicShift(4.5, 0.5, 0, 2),
                    # radial (XY) around pipe corners
                    CyclicShift(4.0, 1.0, 0, 1),
                    CyclicShift(5.0, 2.0, 0, 1),
                    # signed wall distance
                    WallDistance(pipe),
                    # 3D wall disance with boxes
                    UnionBoundaryDistance(boxes),
                    # 1D RBF on X and Y at corner locations
                    RBFShift(4.0, width=10.0, index=0),
                    RBFShift(1.0, width=10.0, index=1),
                    RBFShift(5.0, width=10.0, index=0),
                    RBFShift(2.0, width=10.0, index=1),
                    # 2D RBF around corners in XY-plane
                    RBF2D((4.0, 1.0), width=5.0),
                    RBF2D((5.0, 2.0), width=5.0),
                ]
            )
        )

    @property
    def num_features(self) -> int:
        return len(self.functions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scalar_feats = torch.stack([f(x) for f in self.functions], dim=1)
        vector_feats = torch.cat([f(x) for f in self.vector_functions], dim=1)
        return torch.cat([scalar_feats, vector_feats], dim=1)


class MultiFeatureNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        use_jit: bool = False,
    ):
        super().__init__()
        if use_jit:
            encoding = torch.jit.script(MultiFeatureEncoding())
            dummy = torch.zeros((1, input_dim))
            num_feats = encoding(dummy).shape[1]
        else:
            encoding = MultiFeatureEncoding()
            num_feats = encoding.num_features + len(encoding.vector_functions)

        layers = [
            encoding,
            nn.Sequential(nn.Linear(num_feats, hidden_layers[0]), GELU()),
        ]
        for h_in, h_out in zip(hidden_layers[:-1], hidden_layers[1:]):
            layers.append(nn.Sequential(nn.Linear(h_in, h_out), GELU()))
        layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
