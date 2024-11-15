import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots


class FaceElementType(Enum):
    MIXED = 0
    LINEAR = 2
    TRIANGULAR = 3
    QUADRILATERAL = 4


class FaceType(Enum):
    MIXED = 0
    INTERIOR = 2
    WALL = 3
    PRESSURE_INLET = 4
    PRESSURE_OUTLET = 5
    SYMMETRY = 7
    PERIODIC_SHADOW = 8
    PRESSURE_FAR_FIELD = 9
    VELOCITY_INLET = 10
    PERIODIC = 12
    FAN = 14
    MASS_FLOW_INLET = 20
    INTERFACE = 24
    PARENT = 31
    OUTFLOW = 36
    AXIS = 37


@dataclass
class Connections:
    indexes: list
    cells: list
    normal: np.ndarray = field(default_factory=lambda: np.array([]))
    middle_point: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class Face:
    zone_id: int
    first_index: int
    last_index: int
    type: FaceType
    element_type: FaceElementType
    connections: List[Connections] = field(default_factory=list)
    points: list = field(default_factory=list)


@dataclass
class Grid:
    points: List[List[float]]
    faces: List[Face]
    dim: int

    def plot_nd(self, plot_normals=True, normals_size=10.0):
        fig = make_subplots()

        scatter_method = {
            2: go.Scatter,
            3: go.Scatter3d,
        }

        plot_indexes = set()

        for face in self.faces:
            plot_indexes_face = set()
            all_points = [[] for _ in range(self.dim)]

            for connection in face.connections:
                if 0 in connection.cells:
                    if plot_normals:
                        points = []
                        for ind in connection.indexes:
                            points.append(self.points[ind])
                            points.append(
                                self.points[ind] + connection.normal * normals_size
                            )
                            points.append(self.points[ind])
                        points = np.array(points)
                    else:
                        points = np.array([self.points[i] for i in connection.indexes])
                    plot_indexes_face.update(connection.indexes)
                    if len(points) == 0:
                        continue
                    if face.element_type != 2:
                        all_points[0].extend(
                            [points[:, 0].tolist()[0], points[:, 0].tolist()[-1]]
                        )
                        all_points[1].extend(
                            [points[:, 1].tolist()[0], points[:, 1].tolist()[-1]]
                        )
                        if self.dim == 3:
                            all_points[2].extend(
                                [points[:, 2].tolist()[0], points[:, 2].tolist()[-1]]
                            )

                    all_points[0].extend(
                        [
                            *points[:, 0].tolist(),
                            None,
                        ]
                    )
                    all_points[1].extend(
                        [
                            *points[:, 1].tolist(),
                            None,
                        ]
                    )
                    if self.dim == 3:
                        all_points[2].extend(
                            [
                                *points[:, 2].tolist(),
                                None,
                            ]
                        )
            if self.dim == 2:
                scatter_inputs = {
                    2: {
                        "x": all_points[0],
                        "y": all_points[1],
                        "mode": "lines",
                    },
                }
            else:
                scatter_inputs = {
                    3: {
                        "x": all_points[0],
                        "y": all_points[1],
                        "z": all_points[2],
                        "mode": "lines",
                    },
                }

            plot_indexes.update(plot_indexes_face)
            fig.add_trace(scatter_method[self.dim](**scatter_inputs[self.dim]))
        fig.show()

    def get_face_by_id(self, id):
        for face in self.faces:
            if face.zone_id == id:
                return face


class GridReader:
    def __init__(self):
        self.re_dimline = re.compile(r"\(2\s(\d)\)")
        self.re_comment = re.compile(r"\(0\s.*")
        self.re_zone_init = re.compile(r"\(10\s\(0\s(\w+)\s(\w+)\s(\d+)\)\)")
        self.re_zone_init_5 = re.compile(r"\(10\s\(0\s(\w+)\s(\w+)\s(\d+)\s(\d+)\)\)")
        self.re_zone = re.compile(r"\(10\s\((\w+)\s(\w+)\s(\w+)\s(\d+)\s(\d)\)(\(|)")
        self.re_face_init = re.compile(r"\(13(\s*)\(0\s+(\w+)\s+(\w+)\s+(0|0 0)\)\)")
        self.re_face = re.compile(
            r"\(13(\s*)\((\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\)(\s*)(\(|)"
        )
        self.re_cells = re.compile(r"\(12.*\((\w+)\s+(\w+)\s+(\w+)\s+(\d+)\s+(\d+)\)\)")
        self.re_cells2 = re.compile(
            r"\(12(\s*)\((\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\)(\s*)(\(|)"
        )
        self.re_parthesis = re.compile(r"(^\s*\)(\s*)|^\s*\)\)(\s*)|^\s*\(\s*)")

        self.cell_map = {}
        self.boundary_cells = {}
        self.zones = {}
        self.points = []
        self.faces = []
        self.dim = 0

    def read_zone_vertices(self, first_index, last_index, ifile):
        pos = ifile.tell()
        line = ifile.readline()
        if not re.search(self.re_parthesis, line):
            ifile.seek(pos)
        for i in range(first_index, last_index + 1):
            line = ifile.readline()
            vertex = [eval(x) for x in line.split()]
            self.points[i - 1] = vertex

    def read_faces(self, zone_id, first_index, last_index, type, element_type, ifile):
        face = Face(
            zone_id=zone_id,
            first_index=first_index,
            last_index=last_index,
            type=FaceType(type),
            element_type=FaceElementType(element_type),
        )
        pos = ifile.tell()
        line = ifile.readline()
        if not re.search(self.re_parthesis, line):
            ifile.seek(pos)

        for i in range(face.first_index, face.last_index + 1):
            line = ifile.readline()
            ln = line.split()
            if face.element_type.value == 0:
                nd = int(ln[0], 16)
                indexes = [int(x, 16) - 1 for x in ln[1 : (nd + 1)]]
                cells = [int(x, 16) for x in ln[(nd + 1) :]]
            else:
                nd = face.element_type.value
                indexes = [int(x, 16) - 1 for x in ln[:nd]]
                cells = [int(x, 16) for x in ln[nd:]]

            face.connections.append(
                Connections(
                    indexes=indexes,
                    cells=cells,
                )
            )

            for index in indexes:
                face.points.append(self.points[index - 1])

        self.faces.append(face)

    def scan_fluent_mesh(self, ifile):
        dim = 0
        while 1:
            line = ifile.readline()
            if len(line) == 0:
                break

            if dim == 0:
                a = re.search(self.re_dimline, line)
                if a:
                    dim = int(a.group(1))
                    self.dim = int(a.group(1))
                    continue
            else:
                a = re.search(self.re_zone_init, line) or re.search(
                    self.re_zone_init_5, line
                )
                if a:
                    first_index, last_index, type = (
                        int(a.group(1)),
                        int(a.group(2), 16),
                        int(a.group(3), 16),
                    )
                    self.points = list(range(first_index, last_index + 1))
                    continue

                a = re.search(self.re_zone, line)
                if a:
                    zone_id, first_index, last_index = (
                        int(a.group(1), 16),
                        int(a.group(2), 16),
                        int(a.group(3), 16),
                    )
                    self.read_zone_vertices(first_index, last_index, ifile)
                    continue

                a = re.search(self.re_face, line)
                b = re.search(self.re_face_init, line)
                if a and not b:
                    zone_id, first_index, last_index, type, element_type = (
                        int(a.group(2), 16),
                        int(a.group(3), 16),
                        int(a.group(4), 16),
                        int(a.group(5), 16),
                        int(a.group(6), 16),
                    )
                    self.read_faces(
                        zone_id, first_index, last_index, type, element_type, ifile
                    )
                    continue

    def generateNormals(self):
        for face in self.faces:
            for connection in face.connections:
                points = [self.points[i] for i in connection.indexes]
                if 0 in connection.cells:
                    if len(points[0]) == 2:
                        first_point = np.array(points[0])
                        second_point = np.array(points[1])
                        direction_vector = first_point - second_point
                        normal = np.array(
                            [direction_vector[1], -direction_vector[0]]
                        ) / np.linalg.norm(direction_vector)
                        connection.normal = normal
                    else:
                        first_point = np.array(points[0])
                        second_point = np.array(points[1])
                        third_point = np.array(points[2])
                        normal = np.cross(
                            first_point - second_point, second_point - third_point
                        )
                        normal = normal / np.linalg.norm(normal)
                        connection.normal = normal
                middle_point = np.array(points).sum(axis=0) / len(points)
                connection.middle_point = middle_point

    def read(self, ifilename) -> Grid:
        file, ext = os.path.splitext(ifilename)
        if ext == ".pt":
            return torch.load(ifilename)

        ifile = open(ifilename, "r")
        self.scan_fluent_mesh(ifile)
        ifile.close()
        self.generateNormals()

        return Grid(self.points, self.faces, self.dim)


def grid_msh_to_pt(filename: str):
    grid_file = GridReader().read(filename + ".msh")
    regions_spec = {
        "inlet_zone": 10,
        "outlet_zone": 11,
        "wall_zone": 12,
        "inner_zone": 2,
    }
    grid_file.zones_names = regions_spec
    torch.save(grid_file, filename + ".pt")
