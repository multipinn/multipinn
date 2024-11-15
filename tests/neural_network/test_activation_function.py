import pytest
import torch

from multipinn.neural_network.activation_function import *


# Test for fused_gelu function
@pytest.mark.parametrize(
    "input_tensor,expected_output",
    [
        (torch.tensor([0.0]), torch.tensor([0.0])),  # Test with zero
        (torch.tensor([1.0]), torch.tensor([0.84134474])),  # Test positive value
        (torch.tensor([-1.0]), torch.tensor([-0.15865526])),  # Test negative value
    ],
)
def test_fused_gelu(input_tensor, expected_output):
    actual_output = fused_gelu(input_tensor)
    assert torch.allclose(actual_output, expected_output, atol=1e-5)


# Test for fused_sin function
@pytest.mark.parametrize(
    "input_tensor,expected_output",
    [
        (torch.tensor([0.0]), torch.tensor([0.0])),  # Test with zero
        (torch.tensor([torch.pi / 2]), torch.tensor([1.0])),  # Test for pi/2
        (torch.tensor([torch.pi]), torch.tensor([0.0])),  # Test for pi
        (torch.tensor([-torch.pi / 2]), torch.tensor([-1.0])),  # Test for -pi/2
        (torch.tensor([3 * torch.pi / 2]), torch.tensor([-1.0])),  # Test for 3pi/2
    ],
)
def test_fused_sin(input_tensor, expected_output):
    actual_output = fused_sin(input_tensor)
    assert torch.allclose(actual_output, expected_output, atol=1e-5)


# Test GELU class
def test_gelu_module():
    gelu_layer = GELU()
    input_tensor = torch.tensor([0.0, 1.0, -1.0])
    expected_output = fused_gelu(input_tensor)

    actual_output = gelu_layer(input_tensor)

    assert torch.allclose(actual_output, expected_output, atol=1e-5)


# Test Sine class
def test_sine_module():
    sine_layer = Sine()
    input_tensor = torch.tensor([0.0, torch.pi / 2, torch.pi])
    expected_output = fused_sin(input_tensor)

    actual_output = sine_layer(input_tensor)

    assert torch.allclose(actual_output, expected_output, atol=1e-5)


if __name__ == "__main__":
    pytest.main()
