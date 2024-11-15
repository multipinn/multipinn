import pytest
import torch
import torch.nn as nn

from multipinn.neural_network.fourier_features import *


@pytest.fixture
def sample_tensor():
    return torch.rand(10, 5)  # Random tensor of shape (10, 5)


@pytest.mark.parametrize(
    "input_dim, encoding_dim, sigma, is_trainable",
    [
        (5, 10, 1.0, False),
        (5, 20, 0.5, True),
    ],
)
def test_fourier_encoding(input_dim, encoding_dim, sigma, is_trainable, sample_tensor):
    # Arrange
    encoding_layer = FourierEncoding(input_dim, encoding_dim, sigma, is_trainable)

    # Act
    output = encoding_layer(sample_tensor)

    # Assert
    assert output.shape == (sample_tensor.size(0), encoding_dim)  # Check dimensions
    assert output.requires_grad == is_trainable  # Ensure grad requirement is correct


def test_fourier_feature_network_forward(sample_tensor):
    # Arrange
    input_dim = 5
    output_dim = 3
    hidden_layers = [10, 8]
    encoding_dim = 10
    model = FourierFeatureNetwork(input_dim, output_dim, hidden_layers, encoding_dim)

    # Act
    output = model(sample_tensor)

    # Assert
    assert output.shape == (sample_tensor.size(0), output_dim)


@pytest.mark.parametrize("xavier_init_mode", ["norm", "uniform"])
def test_xavier_initialization(xavier_init_mode, sample_tensor):
    # Arrange
    input_dim = 5
    output_dim = 3
    hidden_layers = [10, 8]
    encoding_dim = 10
    model = FourierFeatureNetwork(
        input_dim,
        output_dim,
        hidden_layers,
        encoding_dim,
        xavier_init=True,
        xavier_init_mode=xavier_init_mode,
    )

    # Act
    output = model(sample_tensor)

    # Assert
    assert output.shape == (sample_tensor.size(0), output_dim)


def test_multiscale_ffnn_forward(sample_tensor):
    # Arrange
    input_dim = 5
    output_dim = 3
    hidden_layers = [12, 10]
    encoding_dim = 10
    sigmas = [0.1, 1.0, 10.0]
    model = MultiScaleFFNN(input_dim, output_dim, hidden_layers, encoding_dim, sigmas)

    # Act
    output = model(sample_tensor)

    # Assert
    expected_output_dim = output_dim
    assert output.shape == (sample_tensor.size(0), expected_output_dim)


def test_forward_with_invalid_input_shape():
    # Arrange
    input_dim = 5
    output_dim = 3
    hidden_layers = [10, 8]
    encoding_dim = 10
    model = FourierFeatureNetwork(input_dim, output_dim, hidden_layers, encoding_dim)
    invalid_input = torch.rand(10, 3)  # Mismatched input dimension

    # Act & Assert
    with pytest.raises(RuntimeError):
        model(invalid_input)  # Should raise a RuntimeError due to dimension mismatch
