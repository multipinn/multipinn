import pytest
from torch.testing import assert_allclose

from multipinn.neural_network.multi_feature_network import *


@pytest.fixture
def sample_input():
    """Fixture to provide sample input tensors."""
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize("shift", [0, 1.5, -1.5])
def test_shift_forward(shift, sample_input):
    """Test the forward method of Shift layer."""
    shift_layer = Shift(shift)
    expected_output = sample_input - shift
    actual_output = shift_layer(sample_input)
    assert_allclose(actual_output, expected_output)


@pytest.mark.parametrize("shift", [0, 1.5, -1.5])
def test_square_shift_forward(shift, sample_input):
    """Test the forward method of SquareShift layer."""
    square_shift_layer = SquareShift(shift)
    expected_output = (sample_input - shift) ** 2
    actual_output = square_shift_layer(sample_input)
    assert_allclose(actual_output, expected_output)


@pytest.mark.parametrize("shift", [0, 1.5, -1.5])
def test_sigmoid_shift_forward(shift, sample_input):
    """Test the forward method of SigmoidShift layer."""
    sigmoid_shift_layer = SigmoidShift(shift)
    expected_output = torch.sigmoid(sample_input - shift)
    actual_output = sigmoid_shift_layer(sample_input)
    assert_allclose(actual_output, expected_output)


@pytest.mark.parametrize(
    "trig_func, scale",
    [
        (torch.sin, 1 / torch.pi),
        (torch.cos, 1 / torch.pi),
        (torch.sin, 1),
        (torch.cos, 1),
    ],
)
def test_scaled_trig_forward(trig_func, scale, sample_input):
    """Test the forward method of ScaledTrig layer."""
    scaled_trig_layer = ScaledTrig(trig_func, scale)
    expected_output = trig_func(torch.pi * sample_input * scale)
    actual_output = scaled_trig_layer(sample_input)
    assert_allclose(actual_output, expected_output)


def test_multi_feature_encoding_num_features():
    """Test the num_features property."""
    encoding = MultiFeatureEncoding()
    assert encoding.num_features == len(encoding.functions)


def test_multi_feature_encoding_forward(sample_input):
    """Test the forward method of MultiFeatureEncoding."""
    encoding = MultiFeatureEncoding()
    actual_output = encoding(sample_input)
    assert actual_output.shape[1] == encoding.num_features * sample_input.shape[1]


@pytest.mark.parametrize("use_jit", [False, True])
def test_multi_feature_network_forward(use_jit, sample_input):
    """Test the forward method of MultiFeatureNetwork."""
    input_dim = sample_input.shape[1]
    output_dim = 3
    hidden_layers = [4, 5]
    network = MultiFeatureNetwork(input_dim, output_dim, hidden_layers, use_jit=use_jit)
    actual_output = network(sample_input)
    assert actual_output.shape == (sample_input.shape[0], output_dim)


def test_multi_feature_network_layers_structure():
    """Test the layers structure of MultiFeatureNetwork."""
    input_dim = 2
    output_dim = 3
    hidden_layers = [4, 5, 3]
    network = MultiFeatureNetwork(input_dim, output_dim, hidden_layers)

    # Layers include encoding and sequential layers
    assert len(network.layers) == len(hidden_layers) + 2
