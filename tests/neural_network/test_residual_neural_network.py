import pytest
import torch
import torch.nn as nn

from multipinn import ResNet


@pytest.fixture
def default_resnet_config():
    input_dim = 10
    output_dim = 5
    hidden_layers = [20, 30, 40]
    return input_dim, output_dim, hidden_layers


@pytest.fixture
def constructed_resnet(default_resnet_config):
    input_dim, output_dim, hidden_layers = default_resnet_config
    return ResNet(
        input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers
    )


def test_resnet_initialization(constructed_resnet):
    assert isinstance(constructed_resnet, nn.Module), "Incorrect instance type"
    assert len(constructed_resnet.layers) > 0, "The number of layers should be non-zero"


def test_layer_dimensions(default_resnet_config):
    input_dim, output_dim, hidden_layers = default_resnet_config
    model = ResNet(input_dim, output_dim, hidden_layers)
    assert (
        model.layers[0][0].in_features == input_dim
    ), "The input layer in_features is incorrect"
    assert (
        model.layers[-1].out_features == output_dim
    ), "The output layer out_features is incorrect"


def test_forward_pass(constructed_resnet, default_resnet_config):
    input_dim, _, _ = default_resnet_config
    sample_input = torch.randn((1, input_dim))
    output = constructed_resnet(sample_input)
    assert output.size(1) == 5, "The output dimension is incorrect"


@pytest.mark.parametrize("blocks", [None, [1], [1, 2], [2, 2, 2]])
def test_custom_blocks(blocks, default_resnet_config):
    input_dim, output_dim, hidden_layers = default_resnet_config
    model = ResNet(input_dim, output_dim, hidden_layers, blocks=blocks)
    # Additional checks can be inserted here as per block configuration specifics


@pytest.mark.parametrize("hidden_layers", [[10], [50], [60, 70]])
def test_edge_case_hidden_layers(hidden_layers):
    input_dim, output_dim = 10, 5
    model = ResNet(
        input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers
    )
    sample_input = torch.randn((1, input_dim))
    try:
        output = model(sample_input)
        assert output.size(1) == output_dim, "Output dimension is incorrect"
    except Exception as e:
        pytest.fail(f"Model failed on edge case inputs: {e}")
