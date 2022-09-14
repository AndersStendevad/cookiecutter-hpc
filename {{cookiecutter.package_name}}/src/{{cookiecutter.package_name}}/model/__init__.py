"""Model
=====
Model related classes and functions used for training and generating.

This submodule contains functions and classes from the following files:

* train.py
* transformer.py

"""

from {{cookiecutter.package_name}}.model.train import train
from {{cookiecutter.package_name}}.model.transformer import Transformer


__all__ = [
    "train",
    "Transformer",
]
