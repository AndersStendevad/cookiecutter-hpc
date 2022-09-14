"""
Loader class for the datastreamer class.
"""
import os
import random

from glob import glob
from abc import ABC, abstractmethod


class Loader(ABC):
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
        Iterate over midi files in self.folder.
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
            for y in glob(os.path.join(x[0], self.extension))
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
