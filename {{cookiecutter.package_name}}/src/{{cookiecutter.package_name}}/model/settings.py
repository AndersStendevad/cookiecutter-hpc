"""Settings
========
Settings used for training and evaluating the model.
"""

from pydantic import BaseSettings


class ModelSettings(BaseSettings):
    """
    Model settings. These are loaded from the .env config file. prefix=MS_

    Parameters
    ----------
    """

    class Config:
        """Config"""

        env_prefix = "MS_"
        env_file = ".env"
        env_file_encoding = "utf-8"
