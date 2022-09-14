"""Preprocess
==========
Preprocessing related classes and functions used for manipulating data.

This includes the Loader. A class for loading files.

This submodule contains functions and classes from the following files:

* loader.py
* dataStreamer.py

"""

from {{cookiecutter.package_name}}.preprocess.loader import Loader
from {{cookiecutter.package_name}}.preprocess.datastreamer import DataStreamer
from {{cookiecutter.package_name}}.preprocess.datastreamer import split

__all__ = [
    "DataStreamer",
    "split",
    "Loader",
]
