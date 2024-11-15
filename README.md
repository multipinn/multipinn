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
