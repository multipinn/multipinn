import pytest
import torch
import torch.nn as nn

from multipinn.neural_network.light_residual_block import LightResidualBlock


@pytest.fixture
def sample_tensors():
    """Fixture to provide sample input tensors for testing."""
    return {
        "small": torch.zeros(3, 5),
        "medium": torch.randn(3, 10),
        "large": torch.randn(3, 50),
    }


@pytest.fixture(params=[nn.ReLU(), nn.Sigmoid(), nn.Tanh()])
def activations(request):
    """Fixture to parametrize different activation functions."""
    return request.param


@pytest.mark.parametrize("features", [5, 10, 50])
def test_light_residual_block_initialization(activations, features):
    """Test initialization of LightResidualBlock with various settings."""
    block = LightResidualBlock(activation=activations, features=features)
    assert isinstance(block, LightResidualBlock)
    assert block.linear_first is not None
    assert block.linear_second is not None


def test_forward_output_shape(activations, sample_tensors):
    """Test the forward method for output shape consistency."""
    for size_label, tensor in sample_tensors.items():
        features = tensor.size(1)
        block = LightResidualBlock(activation=activations, features=features)
        output = block(tensor)
        assert (
            output.shape == tensor.shape
        ), f"Output shape mismatch for {size_label} input"


def test_forward_output_not_none(activations, sample_tensors):
    """Ensure the forward method does not produce None outputs."""
    for tensor in sample_tensors.values():
        features = tensor.size(1)
        block = LightResidualBlock(activation=activations, features=features)
        output = block(tensor)
        assert output is not None, "Output is None, expected Tensor"


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_forward_batch_size_invariance(batch_size, activations):
    """Test that different batch sizes are processed correctly."""
    features = 10
    tensor = torch.randn(batch_size, features)
    block = LightResidualBlock(activation=activations, features=features)
    output = block(tensor)
    assert output.shape == tensor.shape, "Output shape mismatch for batch size"
