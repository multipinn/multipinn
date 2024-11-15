import pytest
import torch
import torch.nn as nn

from multipinn import Inference


class MockModel(nn.Module):
    def __init__(self, input_dim, output_dim=10):
        super(MockModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def mock_model():
    # Creating a mock model for testing
    input_dim = 5
    return MockModel(input_dim)


@pytest.fixture
def inputs():
    return torch.randn(100, 5)  # 100 samples, 5 features per sample


@pytest.fixture
def small_inputs():
    return torch.randn(50, 5)  # Smaller batch for testing different sizes


@pytest.fixture
def inference(mock_model):
    return Inference(model=mock_model, batchsize=32)


@pytest.mark.parametrize("batchsize", [1, 16, 32, 64])
def test_inference_batch_sizes(mock_model, inputs, batchsize):
    inference = Inference(model=mock_model, batchsize=batchsize)
    output = inference(inputs)
    assert output.shape == (inputs.shape[0], mock_model.linear.out_features)


def test_inference_device_handling(mock_model, inputs):
    # Test if it can handle moving tensors to different devices
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    mock_model.to(device)
    inputs = inputs.to(device)

    inference = Inference(model=mock_model)
    output = inference(inputs)

    assert output.device == device
    assert output.shape == (inputs.shape[0], mock_model.linear.out_features)


def input_too_large_dimension(mock_model):
    # Test with input that has incompatible dimensions
    inference = Inference(model=mock_model, batchsize=32)
    large_dim_inputs = torch.randn(10, 20)  # Incompatible input size
    with pytest.raises(RuntimeError):
        inference(large_dim_inputs)
