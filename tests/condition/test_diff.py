import pytest
import torch
from torch import nn

from multipinn.condition.diff import *
from multipinn.condition.diff import _diff_residual


@pytest.fixture
def simple_model():
    # A simple linear model for testing
    return nn.Linear(10, 10, bias=False)


@pytest.fixture
def tensor_args():
    # Fixture for providing common tensor arguments
    return torch.randn((3, 10), requires_grad=True), torch.randn(
        (3, 10), requires_grad=True
    )


def test_unpack():
    # Test unpack function
    batch = torch.randn((3, 10))
    unbound = unpack(batch)
    assert len(unbound) == batch.shape[1]  # Ensure correct number of outputs
    for u in unbound:
        assert u.shape == (3,)  # Confirm each element has expected shape


def test_grad(tensor_args):
    # Test grad function
    arg, _ = tensor_args
    model = nn.Linear(arg.size(1), 1, bias=False)
    u = model(arg)

    gradient = grad(u, arg)
    assert gradient.shape == arg.shape  # Ensure the grad has correct shape


def test_num_diff(simple_model, tensor_args):
    # Test numerical differentiation
    arg, f = tensor_args
    model = simple_model
    direction = torch.randn_like(arg)
    result = num_diff(model, f, arg, direction)
    assert result.shape == (3, 10)  # Ensure expected shape for num_diff result


def test_num_diff_random(simple_model, tensor_args):
    # Test random numerical differentiation
    arg, f = tensor_args
    model = simple_model
    direction = torch.randn_like(arg)
    result = num_diff_random(model, f, arg, direction)
    assert result.shape == (3, 10)  # Ensure shape consistency


def test_num_laplace(simple_model, tensor_args):
    # Test numerical Laplace operator
    arg, f = tensor_args
    model = simple_model
    result = num_laplace(model, f, arg)
    assert result.shape == (3, 10)  # Ensure expected shape for num_laplace


def test_diff_residual(simple_model, tensor_args):
    # Test differential residue
    arg, _ = tensor_args
    model = simple_model
    result = _diff_residual(model, arg)
    assert (
        len(result) == model.in_features * model.out_features
    )  # Ensure correct length of results
    for r in result:
        assert r.shape == (3,)  # Ensure each component has expected shape
