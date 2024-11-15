from unittest import mock

import numpy as np
import pytest
import torch

from multipinn.utils.device import set_device, set_device_and_seed, set_seed


@pytest.fixture
def mock_logger():
    with mock.patch("multipinn.utils.device.logger") as logger:
        yield logger


def test_set_seed():
    seed = 42
    set_seed(seed)
    # Check if seeds were set correctly
    assert torch.initial_seed() == seed
    assert np.random.get_state()[1][0] == seed  # The seed should be in the random state


def test_set_device_cpu(mock_logger):
    with mock.patch("torch.cuda.is_available", return_value=False):
        set_device()
        mock_logger.info.assert_called_once_with(
            mock.ANY
        )  # Device info should have been logged

        assert torch.tensor(0).device.type == "cpu"


def test_set_device_cuda(mock_logger):
    # Create a mock tensor with cuda device
    mock_tensor = mock.MagicMock()
    mock_tensor.device.type = "cuda"

    # Mock necessary torch functions
    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
        "torch.set_default_tensor_type"
    ) as mock_set_tensor_type, mock.patch(
        "torch.tensor", return_value=mock_tensor
    ), mock.patch(
        "torch.get_default_dtype", return_value=torch.float32
    ):
        set_device(gpu_id=0)

        # Verify logger was called
        mock_logger.info.assert_called_once_with(mock.ANY)

        # Verify default dtype
        assert str(torch.get_default_dtype()) == "torch.float32"

        # Verify tensor device
        test_tensor = torch.tensor(0)
        assert test_tensor.device.type == "cuda"

        # Verify tensor type was set correctly
        mock_set_tensor_type.assert_called_once_with("torch.cuda.FloatTensor")


@pytest.mark.parametrize(
    "accelerator, gpu_id, expected_device",
    [
        (None, 0, "cpu"),  # Assuming CUDA is not available
        (None, 0, "cuda:0"),  # Assuming CUDA is available
        ("cuda:0", 1, "cuda:0"),  # Explicitly setting CUDA
    ],
)
def test_set_device_parametric(accelerator, gpu_id, expected_device, mock_logger):
    cuda_available = expected_device != "cpu"

    # Create a context that patches both cuda availability and tensor type setting
    with mock.patch("torch.cuda.is_available", return_value=cuda_available), mock.patch(
        "torch.set_default_tensor_type"
    ) as mock_set_tensor_type:
        set_device(accelerator, gpu_id)

        # Verify the logged device
        logged_device = str(mock_logger.info.call_args[0][0]).split()[-1]
        assert logged_device == expected_device

        # Verify tensor type setting was called correctly
        if cuda_available:
            mock_set_tensor_type.assert_called_once_with("torch.cuda.FloatTensor")
        else:
            mock_set_tensor_type.assert_not_called()


@pytest.mark.parametrize("seed", [42, 0, 1000])
def test_set_device_and_seed(seed):
    set_device_and_seed(seed)
    assert torch.initial_seed() == seed
    assert np.random.get_state()[1][0] == seed  # Check if numpy's seed was also set
