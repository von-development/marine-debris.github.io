from setuptools import setup, find_packages

setup(
    name="marine_ml",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "pytorch-lightning>=1.5.0",
        "wandb>=0.12.0",
        "albumentations>=0.5.2",
        "ultralytics>=8.0.0",
        "scikit-learn>=0.24.0",
        "numpy>=1.19.0",
        "tensorboard>=2.5.0",
        "tensorboardX>=2.4",
        "lightning[extra]>=2.0.0"
    ]
) 