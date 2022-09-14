"""MusicDataset class."""
import pandas as pd
from torch.utils.data import Dataset


class MusicDataset(Dataset):
    def __init__(self, musicfile, transform=None):
        """
        Initialize the MusicDataset.

        Parameters
        ----------
        musicfile : str
            Path to the music file.

        transform : List[Callable], optional
            List of transformations to apply to the data.
        """
        self.music_frame = pd.read_csv(musicfile, header=None)
        self.transform = transform

    def __len__(self):
        """Return the length of the dataset."""
        return self.music_frame.shape[0]

    def __getitem__(self, idx):
        """Return the item at the given index."""
        file_name = self.music_frame.iloc[idx, 0]
        with open(file_name, "r") as f:
            song = [e for t in f.readlines() for e in [t.strip().split(",")]]
        if self.transform:
            return self.transform(song)
        return song
