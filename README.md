# MULTIPINN

## Overview
This project implements a Physics-Informed Neural Network (PINN) for solving differential equations. It provides a flexible framework for various physics-based problems, including fluid dynamics, heat transfer, and more.

## Features
- Support for various neural network architectures (FNN, ResNet, Fourier Features, ...)
- Adaptive point generation for efficient training
- Visualization tools for results analysis
- Integration with Hydra for easy configuration management

## Installation
To install the project, use this command from makefile:

```bash
make install
```

For development purposes, you can install additional dependencies:

```bash
make install.dev
```

### Optional: Graph Neural Network (GNN) Support

If you plan to use Graph Neural Networks (GNNs), you will need to install additional dependencies for `torch-geometric`, which depend on your CUDA version and Python environment. Follow the instructions below to install the necessary packages.

1. **Check Your CUDA Version**:
   Ensure you know your CUDA version. You can check this by running:

   ```bash
   nvcc --version
   ```

   Alternatively, you can check the version of PyTorch's CUDA support:

   ```python
   import torch
   print(torch.version.cuda)
   ```

2. **Install `torch-geometric`**:
   Follow the installation instructions from the [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#optional-dependencies). The installation command will depend on your CUDA version and Python environment. For example:

   ```bash
   # For CUDA 11.8 and torch 2.5.0
    pip install torch_geometric

    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html

   ```

   Replace `cu113` with your specific CUDA version (e.g., `cu102` for CUDA 10.2).

3. **Verify Installation**:
   After installation, verify that `torch-geometric` is correctly installed by running:

   ```python
   import torch_geometric
   print(torch_geometric.__version__)
   ```


## Documentation

Comprehensive documentation is available in the `docs/` directory. You can build the documentation using MkDocs:

```bash
mkdocs build
```

## Testing
To run the tests, use:
```bash
make test
```

## Contributing

Contributions are welcome! Please follow these steps before submitting your pull request:
1. Install pre-commit hooks:

```bash
make install-pre-commit
```
2. Make your changes and commit your code.
