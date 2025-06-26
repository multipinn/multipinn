import torch
from torch_geometric.data import Data

from ..mesh.mesh_domain import MeshDomain
from .condition import Condition


class GraphCondition(Condition):
    """A class representing graph-based boundary or initial conditions for PDE solving.

    This class extends the base Condition class by adding graph-specific functionality
    for handling mesh domains and computing residuals on graph structures.

    Parameters
    ----------
    condition : Condition
        Base condition object containing geometry and function definitions
    mesh : MeshDomain
        Mesh domain object containing the graph structure
    i : int
        Index of the condition in the mesh domain's condition list

    Attributes
    ----------
    mesh_domain : MeshDomain
        Reference to the mesh domain containing the graph structure
    cond_ind : int
        Index of this condition in the mesh domain's condition list
    """

    def __init__(self, condition: Condition, mesh: MeshDomain, i: int):
        self.__dict__.update(condition.__dict__)
        self.mesh_domain = mesh
        self.cond_ind = i

    def select_batch(self, i: int) -> None:
        """Select a specific batch of points for processing.

        This method is not implemented in the base class and should be overridden
        by subclasses if batching functionality is needed.

        Parameters
        ----------
        i : int
            Batch index to select

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        pass

    def set_batching(self, num_batches: int) -> None:
        """Configure the batching settings for this condition.

        This method is not implemented in the base class and should be overridden
        by subclasses if batching functionality is needed.

        Parameters
        ----------
        num_batches : int
            Number of batches to split the data into

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        pass

    def get_residual(self, model):
        """Compute the residual for this condition using the provided model.

        Applies the condition function to the mesh nodes that satisfy this condition
        using the given model.

        Parameters
        ----------
        model : torch.nn.Module
            Neural network model to evaluate

        Returns
        -------
        list
            List of tensors containing the residual values for nodes satisfying
            this condition
        """
        slice = self.mesh_domain.cond_mask[self.cond_ind]

        outputs = self.function(self.mesh_domain.mesh, model)
        outputs_slice = [out * slice for out in outputs]
        return outputs_slice

    def get_residual_fn(self, model):
        """Create a function that computes residuals for arbitrary input points.

        Parameters
        ----------
        model : torch.nn.Module
            Neural network model to evaluate

        Returns
        -------
        callable
            Function that takes point coordinates as input and returns residual values
            after constructing a graph and evaluating the model
        """

        def residual(arg):
            edges = self.mesh_domain.gen_edges(arg)
            arg = Data(nodes=arg, edge_index=edges)
            model.infer_model(arg)
            return self.function(arg, model)

        return residual

    def init_output_len(self, model) -> None:
        """Initialize the output length parameter by evaluating the model on sample points.

        Generates sample points within the geometry's bounding box and computes
        the length of the output vector produced by the residual function.

        Parameters
        ----------
        model : torch.nn.Module
            Neural network model to evaluate

        Notes
        -----
        This method sets the output_len attribute of the condition object.
        """
        # Get device from model parameters
        device = next(model.parameters()).device
        arg_point = (
            torch.tensor(self.geometry.bbox[0], device=device) + torch.tensor(self.geometry.bbox[1], device=device)
        ) * 0.5
        nodes = arg_point.repeat(3, 1) * torch.rand(3, 1, device=device).requires_grad_()
        self.output_len = len(self.get_residual_fn(model)(nodes))
