import os
from glob import glob
from pathlib import Path
from abc import ABC, abstractmethod

import math


import {{cookiecutter.package_name}}.preprocess as preprocess
import sys
import csv


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

    @abstractmethod
    def raw_preprocess(self, transforms=[]):
        """
        Streams raw data.

        Parameters
        ----------
        transforms : list, optional
            List of transforms to be applied to the data. The default is no transforms.
        """

        loader = preprocess.Loader(
            os.path.join(self.data_path, "rawdata"), shuffle=self.shuffle
        )
        transformer = Transformer(transforms)
        n = 0
        filenames = loader.filenames
        sys.stdout.write("Pre-Streaming. {}/{} Done  \r".format(n, len(filenames)))
        sys.stdout.flush()
        for content, name in zip(loader, filenames):

            if content is None:
                n += 1
                sys.stdout.write(
                    "Pre-Streaming. {}/{} Done  \r".format(n, len(filenames))
                )
                sys.stdout.flush()
                continue

            contents = transformer.transform(content)

            if contents is None:
                n += 1
                sys.stdout.write(
                    "Pre-Streaming. {}/{} Done  \r".format(n, len(filenames))
                )
                sys.stdout.flush()
                continue

            for i, x in enumerate(contents):
                content_path = os.path.join(
                    self.data_path, "predata", f"{i}_{Path(name).name}"
                )
                x.save(content_path)

            n += 1
            sys.stdout.write("Pre-Streaming. {}/{} Done  \r".format(n, len(filenames)))
            sys.stdout.flush()
        print("")

    @abstractmethod
    def preprocess(self, transforms=[]):
        """
        Streams data.

        Parameters
        ----------
        transforms : list, optional
            List of transforms to be applied to the data. The default is no transforms.
        """
        loader = preprocess.Loader(
            os.path.join(self.data_path, "predata"), shuffle=self.shuffle
        )
        transformer = Transformer(transforms)
        n = 0
        filenames = loader.filenames

        sys.stdout.write("Streaming. {}/{} Done  \r".format(n, len(filenames)))
        sys.stdout.flush()

        for content, name in zip(loader, filenames):

            if content is None:
                n += 1
                sys.stdout.write("Streaming. {}/{} Done  \r".format(n, len(filenames)))
                sys.stdout.flush()
                continue

            try:
                content = transformer.transform(content)
            except ValueError:
                n += 1
                sys.stdout.write("Streaming. {}/{} Done  \r".format(n, len(filenames)))
                sys.stdout.flush()
                continue

            with open(
                os.path.join(
                    self.data_path, "cleandata", f"{Path(name).name[:-4]}.csv"
                ),
                "w",
            ) as f:
                if type(song[0]) == str:
                    f.write(",".join(content))
                else:
                    writer = csv.writer(f)
                    writer.writerows(content)

            n += 1
            sys.stdout.write("Streaming. {}/{} Done  \r".format(n, len(filenames)))
            sys.stdout.flush()
        print("")

    def ensure_folders(self):
        """
        Ensures the necessary folders exist. Data should already be downlaoded
        """

        if not os.path.exists(os.path.join(self.data_path, "predata")):
            os.mkdir(os.path.join(self.data_path, "predata"))

        if not os.path.exists(os.path.join(self.data_path, "cleandata")):
            os.mkdir(os.path.join(self.data_path, "cleandata"))


class Transformer:
    """
    transform class is used to transform data.
    """

    def __init__(self, transforms=[]):
        """
        Initialize the transform class.
        """

        self.transforms = transforms

    def transform(self, content):
        """
        Returns a song with transformations applied

        Parameters
        ----------
        content : list
            content to be transformed.

        Returns
        -------
        content : list   
            Transformed content.
        """

        for trnsfrm in self.transforms:
            content = trnsfrm(content)
        return cont


def savecontentlist(contentlist, filename):
    with open(filename, "w+") as f:
        for item in contentlist:
            f.write("{}.csv\n".format(item[:-4]))


def split(data_path, train=0.8, val=0.1, test=0.1):

    filenames = [
        y
        for x in os.walk(os.path.join(data_path, "cleandata"))
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
