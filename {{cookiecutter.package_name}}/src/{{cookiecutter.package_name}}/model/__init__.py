"""Model
=====
Model related classes and functions used for training and generating music.

This submodule contains functions and classes from the following files:

* train.py
* trainer.py
* transformer.py
* generate.py
* setting.py
* vocab.py

"""

from {{cookiecutter.package_name}}.model.train import train
from {{cookiecutter.package_name}}.model.trainer import Trainer
from {{cookiecutter.package_name}}.model.transformer import NystromTransformer
from {{cookiecutter.package_name}}.model.generate import generate
from {{cookiecutter.package_name}}.model.settings import ModelSettings, TrainerSettings, GenerateSettings
from {{cookiecutter.package_name}}.model.vocab import Vocab


__all__ = [
    "train",
    "generate",
    "ModelSettings",
    "Vocab",
    "TrainerSettings",
    "GenerateSettings",
    "NystromTransformer",
    "Trainer",
]
