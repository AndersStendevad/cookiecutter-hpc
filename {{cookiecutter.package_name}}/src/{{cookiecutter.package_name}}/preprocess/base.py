"""
Base classes for preprocessing.
"""

import os
import random
import sys

from os.path import exists
from glob import glob
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """
    Class for loading and streaming files.
    """

    def __init__(self, folder, shuffle=False):
        """
        Initialize the Loader class.

        Parameters
        ----------
        folder: str
            Path to folder containing files.
        shuffle: bool
            If True, shuffle files before loading.
        """

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder} does not exist")
        self.folder = folder

        if shuffle:
            self.data = self.shuffled_files
        else:
            self.data = self.files

    @abstractmethod
    def __iter__(self):
        """
        Iterate over files in self.folder.
        """
        ...

    @property
    @abstractmethod
    def extension(self):
        """
        Set file extension

        Returns
        -------
        extension: str
            The extension file to look for
        """
        ...

    @property
    def files(self):
        """
        List of files in self.folder.
        Returns
        -------
        list
            List of filenames in self.folder.
        """
        result = [
            y
            for x in os.walk(self.folder)
            for y in glob(os.path.join(x[0], "*" + self.extension))
        ]

        return result

    @property
    def shuffled_files(self):
        """
        Shuffle midi files in self.folder.

        Returns
        -------
        files: list
            Shuffled list of filenames in self.folder.
        """
        files = self.files
        random.shuffle(files)
        return files


class BaseDataStreamer(ABC):
    """
    DataStreamer class is used to stream data from midi files to event-based representation.
    """

    def __init__(self, data_path, dataset, shuffle=False):
        """
        Initialize the DataStreamer class.

        Parameters
        ----------
        data_path : str
            Path to the data folder.
        shuffle : bool
            Whether to shuffle the data. The default is False.
        """

        self.data_path = data_path
        self.shuffle = shuffle

        self.ensure_folders()

    def ensure_folders(self) -> None:
        """
        Ensures the necessary folders exist. Data should already be downloaded
        """

        if not exists(self.target_folder):
            os.mkdir(self.target_folder)

    def preprocess(self, transforms=[]) -> None:
        """
        Streams data.

        Parameters
        ----------
        transforms : list
            List of transforms to be applied to the data. The default is no transforms.
        """
        loader = self.Loader(self.data_path, shuffle=self.shuffle)
        filenames = loader.files
        n = 0
        sys.stdout.write(f"Streaming: {n}/{len(filenames)} Done\r")
        sys.stdout.flush()
        for filename in loader:
            self.transform(filename)
            n += 1
            sys.stdout.write(f"Streaming: {n}/{len(filenames)} Done\r")
            sys.stdout.flush()
        print("")

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
    def Loader(self) -> BaseLoader:
        """
        Returns
        ----------
        Loader : BaseLoader
            Loader object to use
        """
        ...

    @abstractmethod
    def transform(self, filename, content=None, transforms=None) -> None:
        """
        Transforms data and saves to disk

        Parameters
        ----------
        filename: str
            Name of file without extension
        content
            content to be transformed.
        transforms : list
            List of transforms to be applied
        """
        ...
