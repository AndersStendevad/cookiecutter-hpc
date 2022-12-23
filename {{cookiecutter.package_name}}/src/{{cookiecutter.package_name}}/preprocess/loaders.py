"""
Loader classes for the datastreamers class.
"""

from os.path import join

from rifs.preprocess.base import BaseLoader


class WavLoader(BaseLoader):
    """Loader used to load files"""

    def __init__(self, data_path, dataset, shuffle):
        """
        Initialize the Loader class.

        Parameters
        ----------
        data_path : str
            Path to the data folder.
        dataset : str
            Name of the dataset.
        shuffle: bool
            If True, shuffle files before loading.
        """
        folder = join(data_path, f"raw_data/{dataset}/audio")
        super().__init__(folder, shuffle)

    def __iter__(self):
        """
        Iterate over midi files in self.folder.
        """
        for filename in self.files:
            yield filename

    @property
    def extension(self):
        """
        Set file extension

        Returns
        -------
        extension: str
            The extension file to look for
        """
        return ".wav"


