from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("requirements-dev.txt") as f:
    dev_requirements = f.read().splitlines()

setup(
    # Basic info
    name="multipinn",
    version="1.0.0",
    packages=find_packages(exclude=("tests*")),
    author="LabADT",
    author_email="chermentsgoev@yandex.ru",
    url="https://mca.nsu.ru/labadt",
    # License and description
    license="LICENSE",
    description="Multitask PINN implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8, <3.12",
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
    zip_safe=False,
)
