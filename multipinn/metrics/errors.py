import torch


def mean_squared_error(true, pred):
    return torch.sqrt(torch.sum((true - pred) ** 2))


def relative_error(true, pred):
    return torch.abs(true - pred) / torch.linalg.norm(true)


def norm_error(true, pred):
    return torch.linalg.norm(true - pred) / torch.linalg.norm(true)


def per_axis_relative_error(true, pred):
    return torch.mean(torch.abs(true - pred), axis=0) / torch.mean(
        torch.abs(true), axis=0
    )


def l_inf_error(true, pred):
    return (true - pred).abs().max()
