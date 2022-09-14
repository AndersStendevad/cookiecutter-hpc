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
    hidden_size : int
        Size of the hidden layer
    num_hidden_layers : int
        Number of hidden layers
    num_attention_heads : int
        Number of attention heads
    intermediate_size : int
        Size of the intermediate layer
    layer_norm_eps : float
        Epsilon value for layer normalization
    """

    hidden_size: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 128
    intermediate_size: int = 1024
    layer_norm_eps: float = 1e-12

    class Config:
        """Config"""

        env_prefix = "MS_"
        env_file = ".env"
        env_file_encoding = "utf-8"
