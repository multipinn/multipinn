import pytest
import torch

from multipinn.neural_network.pirate_net import *


# Fixtures for mock data
@pytest.fixture
def random_tensor():
    return torch.randn(8, 256)  # Batch size of 8, hidden_size/input_dim of 256


@pytest.fixture
def bottleneck_module():
    return PIModifiedBottleneck(hidden_size=256, output_dim=256, nonlinearity=0.5)


@pytest.fixture
def pirate_net_module():
    return PirateNet(
        input_dim=256,
        output_dim=10,
        hidden_size=256,
        num_blocks=3,
        nonlinearity=0.0,
        gamma=1.0,
    )


# Test PIModifiedBottleneck
def test_bottleneck_forward(bottleneck_module, random_tensor):
    u = torch.randn(8, 256)
    v = torch.randn(8, 256)
    output = bottleneck_module(random_tensor, u, v)

    assert output.shape == (8, 256), "Output shape mismatch for PIModifiedBottleneck"


def test_bottleneck_learnable_parameter(bottleneck_module):
    alpha = bottleneck_module.alpha
    assert alpha.requires_grad, "Alpha should be a learnable parameter"


# Test PirateNet
def test_pirate_net_forward(pirate_net_module, random_tensor):
    output = pirate_net_module(random_tensor)

    assert output.shape == (8, 10), "Output shape mismatch for PirateNet"


def test_pirate_net_initialization(pirate_net_module):
    assert len(pirate_net_module.blocks) == 3, "The number of blocks should match 3"
    assert pirate_net_module.input_dim == 256, "Input dimension should match 256"
    assert pirate_net_module.output_dim == 10, "Output dimension should match 10"


# Parametric test for input size validation in PirateNet
@pytest.mark.parametrize("invalid_input", [torch.randn(8, 128), torch.randn(8, 512)])
def test_pirate_net_invalid_input(pirate_net_module, invalid_input):
    with pytest.raises(RuntimeError):
        pirate_net_module(invalid_input)
