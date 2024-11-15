import pytest
import torch
import torch.nn as nn

from multipinn import FNN, XavierFNN


@pytest.fixture
def input_dim():
    return 4


@pytest.fixture
def output_dim():
    return 2


@pytest.fixture
def hidden_layers():
    return [8, 6, 4]


@pytest.fixture
def sample_input(input_dim):
    return torch.randn(3, input_dim)  # Batch size of 3


def test_fnn_forward_pass(input_dim, output_dim, hidden_layers, sample_input):
    model = FNN(input_dim, output_dim, hidden_layers)
    output = model(sample_input)
    assert output.shape == (3, output_dim)  # Check output shape


def test_fnn_layer_initialization(input_dim, output_dim, hidden_layers):
    model = FNN(input_dim, output_dim, hidden_layers)
    assert len(model.network_layers) == len(hidden_layers) + 1  # Verify layer count
    assert isinstance(
        model.network_layers[0][1], nn.Module
    )  # Check that the first activation is Sine
    assert isinstance(
        model.network_layers[-2][1], nn.Module
    )  # Check that the second to last is GELU
    assert isinstance(
        model.network_layers[-1], nn.Linear
    )  # Check that the last layer is Linear


@pytest.mark.parametrize("init_mode", ["uniform"])
def test_xavier_fnn_initialization(input_dim, output_dim, hidden_layers, init_mode):
    model = XavierFNN(input_dim, output_dim, hidden_layers, init_mode=init_mode)

    for layer in model.network_layers[:-1]:
        weight = layer[0].weight
        bias = layer[0].bias
        assert weight.min().item() <= 0.0 and weight.max().item() >= 0.0
        assert torch.all(bias == 0).item()

    output_layer = model.network_layers[-1]
    weight = output_layer.weight
    bias = output_layer.bias
    assert weight.min().item() <= 0.0 and weight.max().item() >= 0.0
    assert torch.all(bias == 0).item()
