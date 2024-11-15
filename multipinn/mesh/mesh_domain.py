import os

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph

from multipinn.geometry.domain import Domain
from multipinn.geometry.shell import BaseShell


class MeshDomain:
    """A class for generating and managing mesh structures with node connectivity.

    This class handles mesh generation with k-nearest neighbors and optional radius-based connectivity.
    It supports both dynamic mesh generation and loading from files.

    Parameters
    ----------
    k : int, optional
        Number of nearest neighbors for generating edges in the graph (default: 5)
    r : float, optional
        Radius for additional edge connections. If provided, edges will be added between
        all points within this radius (default: None)

    Attributes
    ----------
    mesh : torch_geometric.data.Data or None
        The generated mesh structure containing nodes and edges
    cond_mask : dict or None
        Dictionary mapping condition indices to boolean masks indicating which nodes
        satisfy each condition
    from_file : bool
        Flag indicating whether the mesh was loaded from files
    """

    def __init__(self, k=5, r=None, mesh=None, cond_mask=None, from_file=None):
        self.k = k
        self.r = r
        self.mesh = mesh
        self.cond_mask = cond_mask
        self.from_file = from_file

    def gen_edges(self, points):
        """Generate edges between mesh points using k-NN and optional radius-based connectivity.

                Parameters
                ----------
                points : torch.Tensor
                    Tensor of point coordinates with shape (n_points, n_dimensions)
        pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
                Returns
                -------
                torch.Tensor
                    Edge indices tensor with shape (2, n_edges) representing the connectivity graph.
                    Each column represents an edge between two nodes.
        """
        edges = knn_graph(points, k=self.k, loop=False)

        if self.r is not None:
            edges_radius = radius_graph(points, r=self.r, loop=False)
            edges = torch.cat([edges, edges_radius], 1)

        edges = torch.unique(edges, dim=0)
        return edges

    def update_mesh(self, conditions):
        """Update the mesh structure based on given conditions.

        Creates a new mesh by concatenating points from all conditions and generating
        appropriate edge connections. Also creates condition masks for each set of points.

        Parameters
        ----------
        conditions : list
            List of condition objects, each containing points attribute representing
            node coordinates for that condition.

        Notes
        -----
        This method updates the following attributes:
        - self.cond_mask : Dictionary of boolean masks for each condition
        - self.mesh : Updated Data object containing the new nodes and edges
        """
        if self.from_file:
            return

        indx = 0
        self.cond_mask = {}
        nodes = torch.cat([cond.points for cond in conditions], 0)

        print(nodes.shape)
        for i, cond in enumerate(conditions):
            mask = torch.zeros(nodes.shape[0], dtype=torch.bool)
            mask[indx : indx + cond.points.shape[0]] = 1
            self.cond_mask[i] = mask
            indx += cond.points.shape[0]
        edges = self.gen_edges(nodes)
        self.mesh = Data(nodes=nodes, edge_index=edges)

    @classmethod
    def from_mesh_files(cls, nodes_path, edges_path, conditions, k):
        nodes = np.loadtxt(nodes_path, delimiter=",", usecols=[0, 1])
        edges = np.loadtxt(edges_path, delimiter=",", dtype=np.int32)

        cond_mask = {}
        for i, cond in enumerate(conditions):
            if isinstance(cond.geometry, Domain):
                mask = torch.tensor(cond.geometry.inside(nodes), dtype=torch.bool)
                cond_mask[i] = mask
            if isinstance(cond.geometry, BaseShell):
                mask = torch.tensor(cond.geometry.on_boundary(nodes), dtype=torch.bool)
                cond_mask[i] = mask

        nodes = torch.tensor(nodes, dtype=torch.float32, requires_grad=True)
        edges = torch.tensor(edges, dtype=torch.int64).T

        return cls(
            k=k,
            r=None,
            mesh=Data(nodes=nodes, edge_index=edges),
            cond_mask=cond_mask,
            from_file=True,
        )
