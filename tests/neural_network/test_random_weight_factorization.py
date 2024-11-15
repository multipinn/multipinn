import pytest
import torch
import torch.nn as nn

from multipinn import FactorizedDense, FactorizedFNN


@pytest.fixture
def random_tensor():
    """Fixture for generating random input tensors"""

    def _random_tensor(shape):
        return torch.randn(*shape)

    return _random_tensor


def test_factorized_dense_initialization():
    # Test different initializations
    layer = FactorizedDense(10, 20)
    assert layer.in_features == 10
    assert layer.out_features == 20
    assert isinstance(layer.s, nn.Parameter)
    assert isinstance(layer.v, nn.Parameter)

    layer_no_bias = FactorizedDense(10, 20, bias=False)
    assert layer_no_bias.bias is None

    assert layer.bias is not None  # Check if bias is properly initialized when expected


def test_factorized_dense_forward(random_tensor):
    layer = FactorizedDense(10, 20)
    input_tensor = random_tensor((5, 10))  # Batch of 5, 10 features each
    output = layer(input_tensor)
    assert output.shape == (5, 20)  # Ensure output shape is correct


@pytest.mark.parametrize(
    "input_dim, output_dim, hidden_layers, input_shape",
    [
        (10, 1, [32, 64], (5, 10)),  # Batch of 5, input dim 10
        (20, 1, [50, 30], (3, 20)),  # Batch of 3, input dim 20
    ],
)
def test_factorized_fnn_forward(
    input_dim, output_dim, hidden_layers, input_shape, random_tensor
):
    model = FactorizedFNN(input_dim, output_dim, hidden_layers)
    input_tensor = random_tensor(input_shape)
    output = model(input_tensor)
    assert output.shape == (input_tensor.shape[0], output_dim)


def test_extra_repr():
    layer = FactorizedDense(10, 20, bias=True)
    repr_str = layer.extra_repr()
    assert "in_features=10" in repr_str
    assert "out_features=20" in repr_str
    assert "bias=True" in repr_str

    layer_no_bias = FactorizedDense(10, 20, bias=False)
    repr_str_no_bias = layer_no_bias.extra_repr()
    assert "bias=False" in repr_str_no_bias


if __name__ == "__main__":
    pytest.main()
