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
            self.songs = self.shuffled_midifiles
        else:
            self.songs = self.midifiles

    def __iter__(self):
        """
        Iterate over midi files in self.folder.
        """
        songs = iter(self.songs)
        while True:
            try:
                yield mido.MidiFile(next(songs), clip=True)
            except StopIteration:
                break
            except (
                IndexError,
                EOFError,
                mido.midifiles.meta.KeySignatureError,
                ValueError,
            ):
                yield None

    @staticmethod
    def _play(midipath):
        """
        Play a midi file.
        Parameters
        ----------
        midipath: str
            Path to midi file.

        Returns
        -------

        """
        if not os.path.exists(midipath):
            raise FileNotFoundError(f"File {midipath} does not exist")

        os.system("xdg-open {}".format(midipath))

    @staticmethod
    def play(mid):
        """
        Play a midi file.
        Parameters
        ----------
        mid: mido.MidiFile
            Midi file to play.

        Returns
        -------

        """
        mid.save("temp.mid")
        MidiLoader._play("temp.mid")

    @property
    def midifiles(self):
        """
        List of midi files in self.folder.
        Returns
        -------
        list
            List of midi filenames in self.folder.
        """
        result = [
            y for x in os.walk(self.folder) for y in glob(os.path.join(x[0], "*.mid"))
        ]

        return result

    @property
    def shuffled_midifiles(self):
        """
        Shuffle midi files in self.folder.

        Returns
        -------
        songs: list
            Shuffled list of midi filenames in self.folder.

        """
        songs = self.midifiles
        random.shuffle(songs)
        return songs

    @property
    def next_song(self):
        """
        Get next song in self.songs.
        Returns
        -------

        """
        return mido.MidiFile(next(self.songs), clip=True)

    @property
    def random_song(self):
        """
        Get random song in self.songs.

        Returns
        -------
        mido.MidiFile
            Random midi file in self.songs.
        """
        return mido.MidiFile(random.choice(self.songs), clip=True)
