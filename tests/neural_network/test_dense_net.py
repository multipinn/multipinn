import pytest

from multipinn.neural_network.dense_net import *


@pytest.fixture
def dense_net_params():
    return {
        "input_dim": 10,
        "output_dim": 2,
        "neurons_per_block": [10, 20],
        "layers_per_block": [3, 2],  # Optional
    }


@pytest.fixture
def dense_block_params():
    return {"layers_amount": 3, "features": 10}


@pytest.fixture
def input_tensor(dense_net_params):
    return torch.rand(5, dense_net_params["input_dim"])  # Batch size of 5


def test_densenet_initialization(dense_net_params):
    net = DenseNet(**dense_net_params)
    assert isinstance(net, DenseNet), "Failed to initialize DenseNet"
    assert len(net.layers) == 4, "Incorrect number of layers in DenseNet"


def test_denseblock_initialization(dense_block_params):
    block = DenseBlock(**dense_block_params)
    assert isinstance(block, DenseBlock), "Failed to initialize DenseBlock"
    assert (
        len(block.layers) == dense_block_params["layers_amount"]
    ), "Incorrect number of layers in DenseBlock"


def test_densenet_forward_pass(input_tensor, dense_net_params):
    net = DenseNet(**dense_net_params)
    output = net(input_tensor)
    assert output.shape == (
        5,
        dense_net_params["output_dim"],
    ), "Output shape mismatch in DenseNet forward pass"


def test_denseblock_forward_pass(input_tensor, dense_block_params):
    block = DenseBlock(**dense_block_params)
    output = block(input_tensor)
    assert (
        output.shape == input_tensor.shape
    ), "Output shape mismatch in DenseBlock forward pass"


@pytest.mark.parametrize(
    "input_dim, output_dim, neurons_per_block, expected_layers",
    [(10, 2, [10, 20, 30], 5), (5, 1, [5, 3], 4), (3, 3, [3, 5, 7, 2], 6)],
)
def test_densenet_various_params(
    input_dim, output_dim, neurons_per_block, expected_layers
):
    net = DenseNet(input_dim, output_dim, neurons_per_block)
    assert (
        len(net.layers) == expected_layers
    ), f"Expected {expected_layers} layers, got {len(net.layers)}"


def test_densenet_invalid_params():
    with pytest.raises(TypeError):
        DenseNet("invalid", 2, [10, 20, 30])  # invalid input_dim type
    with pytest.raises(IndexError):
        DenseNet(10, 2, [], layers_per_block=[1, 2])  # mismatched block sizes


@pytest.mark.parametrize("layers_amount, features", [(3, 10), (1, 5), (5, 20)])
def test_denseblock_multiple_initializations(layers_amount, features):
    block = DenseBlock(layers_amount, features)
    assert isinstance(block, DenseBlock), "Failed to initialize DenseBlock"
    assert (
        len(block.layers) == layers_amount
    ), "Incorrect number of layers in DenseBlock"
