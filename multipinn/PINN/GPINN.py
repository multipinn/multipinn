from typing import List

import torch

from ..condition import Condition, GraphCondition
from ..mesh.mesh_domain import MeshDomain
from .PINN import PINN


class GPINN(PINN):
    """Graph-based Physics-Informed Neural Network for solving PDEs.

    This class extends the base PINN class by incorporating graph-based methods
    for solving partial differential equations using neural networks on mesh structures.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model that will be used for solving the PDE
    conditions : List[Condition]
        List of boundary and initial conditions for the PDE
    mesh : MeshDomain
        Mesh domain object that defines the spatial discretization and connectivity

    Attributes
    ----------
    model : torch.nn.Module
        Neural network model with added mesh attribute
    mesh_domain : MeshDomain
        Reference to the mesh domain containing the graph structure
    conditions : List[GraphCondition]
        List of graph-adapted conditions created from the input conditions

    Notes
    -----
    During initialization, the class:
    - Converts all conditions to GraphConditions
    - Attaches the mesh to the model
    - Initializes output lengths for all conditions
    - Sets the model to evaluation mode
    """

    def __init__(
        self,
        model,
        conditions: List[Condition],
        mesh: MeshDomain,
    ) -> None:
        conditions = [
            GraphCondition(cond, mesh, i) for i, cond in enumerate(conditions)
        ]
        self.mesh_domain = mesh

        model.mesh = self.mesh_domain

        super().__init__(model, conditions)

    def update_data(self) -> None:
        """Update the computational domain and mesh structure.

        Updates the points in all conditions and regenerates the mesh structure
        based on the updated points. This method should be called whenever the
        spatial discretization needs to be modified.

        Notes
        -----
        This method performs two main operations:
        1. Updates points for each condition using the current model
        2. Regenerates the mesh structure based on the updated points
        """
        for cond in self.conditions:
            cond.update_points(model=self.model)
        self.mesh_domain.update_mesh(self.conditions)

    def calculate_loss(self) -> List[torch.Tensor]:
        """Calculate the total loss for all conditions.

        Computes residuals and losses for all boundary and initial conditions
        using the current model state and mesh structure.

        Returns
        -------
        List[torch.Tensor]
            List of loss tensors for each component of all conditions

        Notes
        -----
        The method:
        1. Performs model inference on the entire mesh
        2. Calculates losses for each condition
        3. Combines all losses into a single list
        """
        mean_losses = []
        self.model.infer_model(self.mesh_domain.mesh)

        for cond in self.conditions:
            loss = self.condition_loss(cond)
            mean_losses += self.mean_loss(loss)
        return mean_losses
