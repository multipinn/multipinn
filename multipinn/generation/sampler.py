from typing import Any

import numpy as np
import skopt


def sample(n_samples: int, dimension: int, sampler: str = "pseudo") -> np.ndarray:
    """
    Generate pseudorandom or quasirandom samples in [0, 1]^dimension.

    Args:
        n_samples (int): The number of samples to generate.
        dimension (int): The dimensionality of the space.
        sampler (str, optional): Sampling strategy to use. Options are:
            - "pseudo": Pseudorandom sampling.
            - "LHS": Latin Hypercube Sampling.
            - "Halton": Halton sequence.
            - "Hammersley": Hammersley sequence.
            - "Sobol": Sobol sequence.
            Defaults to "pseudo".

    Returns:
        np.ndarray: A (n_samples, dimension) array of generated points.
    """
    if sampler == "pseudo":
        return _pseudorandom(n_samples, dimension)
    return _quasirandom(n_samples, dimension, sampler)


def _pseudorandom(n_samples: int, dimension: int) -> np.ndarray:
    """
    Generate pseudorandom samples in [0, 1]^dimension.

    Args:
        n_samples (int): The number of samples to generate.
        dimension (int): The dimensionality of the space.

    Returns:
        np.ndarray: A (n_samples, dimension) array of pseudorandom points.
    """
    return np.random.random(size=(n_samples, dimension)).astype("float32")


def _quasirandom(n_samples: int, dimension: int, sampler_name: str) -> np.ndarray:
    """
    Generate quasirandom samples using specified sampling strategy.

    Args:
        n_samples (int): The number of samples to generate.
        dimension (int): The dimensionality of the space.
        sampler_name (str): The name of the quasirandom sampler to use.

    Returns:
        np.ndarray: A (n_samples, dimension) array of quasirandom points.

    Raises:
        ValueError: If the specified sampler_name is not supported.
    """
    sampler = _get_sampler(sampler_name)
    space = [(0.0, 1.0)] * dimension
    result = np.asarray(sampler.generate(space, n_samples), dtype="float32")

    if sampler_name == "LHS":
        # LHS is already randomized, no further processing needed
        return result
    else:
        return _extra_randomization(result)


def _get_sampler(sampler_name: str) -> Any:
    """
    Retrieve the sampler object based on the sampler name.

    Args:
        sampler_name (str): The name of the sampler.

    Returns:
        Any: An instance of the sampler.

    Raises:
        ValueError: If the sampler_name is not recognized.
    """
    if sampler_name == "LHS":
        return skopt.sampler.Lhs()
    elif sampler_name == "Halton":
        return skopt.sampler.Halton(min_skip=1, max_skip=1_000_000)
    elif sampler_name == "Hammersley":
        return skopt.sampler.Hammersly(min_skip=1, max_skip=1_000_000)
    elif sampler_name == "Sobol":
        return skopt.sampler.Sobol(randomize=True)
    else:
        raise ValueError(f"'{sampler_name}' sampling is not available.")


def _extra_randomization(quasirandom_arr: np.ndarray) -> np.ndarray:
    """
    Apply extra randomization to quasirandom samples to avoid boundary artifacts.

    Args:
        quasirandom_arr (np.ndarray): Array of quasirandom samples.

    Returns:
        np.ndarray: Randomized array of samples.
    """
    samples, dim = quasirandom_arr.shape
    sigma = samples ** (-1 / dim) / 8
    shift = np.random.uniform(size=(1, dim))

    quasirandom_arr += np.random.randn(samples, dim) * sigma + shift
    return quasirandom_arr % 1
