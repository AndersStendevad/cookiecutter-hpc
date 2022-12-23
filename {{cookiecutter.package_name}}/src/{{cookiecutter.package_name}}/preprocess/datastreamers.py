"""
DataStreamers used to stream data
"""


import os
import wave
import pandas as pd

from glob import glob
from os.path import join
from pathlib import Path

from rifs.preprocess import Loader
from rifs.preprocess.base import BaseDataStreamer


class WavDataStreamer(BaseDataStreamer):
    """WavDataStreamer used to stream data"""

    def __init__(self, data_path, dataset, shuffle=False):
        """
        Initialize the WavDataStreamer class.

        Parameters
        ----------
        data_path : str
            Path to the data folder.
        dataset : str
            Name of the dataset.
        shuffle : bool
            Whether to shuffle the data. The default is False.
        """
        dataframes = []
        for filename in glob(join(data_path, f"raw_data/{dataset}/text", "*.tsv")):
            dataframes.append(pd.read_csv(filename, sep="\t"))
        self.utterances = pd.concat(dataframes)
        self.utterances["meeting_id"] = self.utterances.apply(
            lambda row: "_".join(row.utterance_id.split("_")[1:3]), axis=1
        )
        super().__init__(data_path, shuffle)

    @property
    def target_folder(self) -> str:
        """
        Returns
        ----------
        target_folder: str
            Name of target folder
        """
        return join(self.data_path, "clean_data")

    @property
    def Loader(self):
        """Loader used to load data"""
        return Loader

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
        meeting_id = Path(filename).stem
        year = Path(filename).parent.stem
        os.makedirs(join(self.target_folder, year, meeting_id), exist_ok=True)
        speeches = self.utterances[
            self.utterances["meeting_id"] == meeting_id
        ].sort_values(by=["start_time"])

        with wave.open(filename, "rb") as infile:
            nchannels = infile.getnchannels()
            sampwidth = infile.getsampwidth()
            framerate = infile.getframerate()

            for speech in speeches.itertuples():
                start = float(speech.start_time)
                end = float(speech.end_time)
                with wave.open(
                    join(
                        self.target_folder,
                        year,
                        meeting_id,
                        speech.utterance_id + ".wav",
                    ),
                    "w",
                ) as outfile:
                    infile.setpos(int(start * framerate))
                    data = infile.readframes(int((end - start) * framerate))
                    outfile.setnchannels(nchannels)
                    outfile.setsampwidth(sampwidth)
                    outfile.setframerate(framerate)
                    outfile.setnframes(int(len(data) / sampwidth))
                    outfile.writeframes(data)
            if transforms:
                raise Exception("This DataStreamer does not support transforms")


