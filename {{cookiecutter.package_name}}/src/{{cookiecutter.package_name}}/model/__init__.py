"""Model
=====
Model related classes and functions used for training and generating.

This submodule contains functions and classes from the following files:

* train.py
* transformer.py
* setting.py

"""

from {{cookiecutter.package_name}}.model.train import train
from {{cookiecutter.package_name}}.model.transformer import NystromTransformer
from {{cookiecutter.package_name}}.model.settings import ModelSettings


__all__ = [
    "train",
    "ModelSettings",
    "NystromTransformer",
]
