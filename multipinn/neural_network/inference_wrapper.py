import torch
import torch.nn as nn


class Inference(nn.Module):
    """
    A wrapper class for model inference that handles batch processing of input data.

    Args:
        model (nn.Module): The neural network model to be used for inference.
        batchsize (int, optional): Size of batches for processing data. Defaults to 256.
    """

    def __init__(self, model: nn.Module, batchsize: int = 256):
        super(Inference, self).__init__()
        self.batchsize = batchsize
        self.model = model
        self.device = next(model.parameters()).device

    def forward(self, x: torch.Tensor):
        """
        Forward pass that processes input data in batches.

        Args:
            x (torch.Tensor): Input tensor to be processed.

        Returns:
            torch.Tensor: Concatenated output tensor containing predictions for all batches.
        """
        x = x.to(self.device)
        x_loader = torch.utils.data.DataLoader(x, self.batchsize)
        all_outputs = []
        for mini_x in x_loader:
            predict = self.model.forward(mini_x)
            all_outputs.append(predict)
        all_outputs = torch.vstack(all_outputs)
        return all_outputs
