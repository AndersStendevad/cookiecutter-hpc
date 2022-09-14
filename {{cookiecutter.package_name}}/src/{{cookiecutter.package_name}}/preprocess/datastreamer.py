"""
Datastreamer
"""

import os
from glob import glob
from abc import ABC, abstractmethod

import math


from {{cookiecutter.package_name}}.preprocess import Loader
import sys


class DataStreamer(ABC):
    """
    DataStreamer class is used to stream data from midi files to event-based representation.
    """

    def __init__(self, data_path, shuffle=False):
        """
        Initialize the DataStreamer class.

        Parameters
        ----------
        data_path : str
            Path to the data folder.
        shuffle : bool, optional
            Whether to shuffle the data. The default is False.
        """

        self.data_path = data_path
        self.shuffle = shuffle

        self.ensure_folders()

    def ensure_folders(self) -> None:
        """
        Ensures the necessary folders exist. Data should already be downloaded
        """

        if not os.path.exists(os.path.join(self.data_path, self.target_folder)):
            os.mkdir(os.path.join(self.data_path, self.target_folder))

    @property
    @abstractmethod
    def target_folder(self) -> str:
        """
        Returns
        ----------
        target_folder: str
            Name of target folder
        """
        ...

    @property
    @abstractmethod
    def Loader(self) -> Loader:
        """
        Returns
        ----------
        Loader : Loader
            Loader object to use
        """
        ...

    @abstractmethod
    def preprocess(self, transforms=[]) -> None:
        """
        Streams data.

        Parameters
        ----------
        transforms : list, optional
            List of transforms to be applied to the data. The default is no transforms.
        """
        loader = self.Loader(
            os.path.join(self.data_path, self.target_folder), shuffle=self.shuffle
        )

        n = 0
        filenames = loader.filenames

        sys.stdout.write("Streaming. {}/{} Done  \r".format(n, len(filenames)))
        sys.stdout.flush()

        for content, filename in zip(loader, filenames):

            content = self.transform(content, filename, transforms)

            n += 1
            sys.stdout.write("Streaming. {}/{} Done  \r".format(n, len(filenames)))
            sys.stdout.flush()

        print("")

    @abstractmethod
    def transform(self, content, filename, transforms) -> None:
        """
        Transforms data and saves to disk

        Parameters
        ----------
        content
            content to be transformed.
        filename: str
            Name of file without extension
        transforms : list
            List of transforms to be applied
        """
        ...


def savecontentlist(contentlist, filename) -> None:
    """
    Saves a list of content to a csv file

    Parameters
    ----------
    contentlist : list
        List of content to be saved.
    filename : str
        Name of file to save to.
    """
    with open(filename, "w+") as f:
        for item in contentlist:
            f.write("{}.csv\n".format(item[:-4]))


def split(data_path, train=0.8, val=0.1, test=0.1) -> None:
    """
    Parameters
    ----------
    data_path : str
        Path to the data folder.
    train : float, optional
        Percentage of data to be used for training. The default is 0.8.
    val : float, optional
        Percentage of data to be used for validation. The default is 0.1.
    test : float, optional
        Percentage of data to be used for testing. The default is 0.1.

    Returns
    -------
    None.
    """

    filenames = [
        y
        for x in os.walk(os.path.join(data_path, "clean_data"))
        for y in glob(os.path.join(x[0], "*.csv"))
    ]
    if train + val + test != 1:
        raise Exception("Set portions don't add up to 1")

    train_portion = math.floor(train * len(filenames))
    val_portion = math.floor(val * len(filenames))
    test_portion = math.floor(test * len(filenames))

    train = filenames[:train_portion]
    val = filenames[train_portion : train_portion + val_portion]  # noqa
    test = filenames[-test_portion:]

    savecontentlist(train, os.path.join(data_path, "train.txt"))
    savecontentlist(val, os.path.join(data_path, "val.txt"))
    savecontentlist(test, os.path.join(data_path, "test.txt"))
