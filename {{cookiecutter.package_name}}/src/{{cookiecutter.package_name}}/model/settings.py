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


class TrainerSettings(BaseSettings):
    """Trainer settings. These are loaded from the .env config file. prefix=TS_

    Parameters
    ----------
    batch_size : int
        Batch size
    lr : float
        Learning rate
    """

    batch_size: int = 64
    lr: float = 1e-7

    class Config:
        """Config"""

        env_prefix = "TS_"
        env_file = ".env"
        env_file_encoding = "utf-8"


class GenerateSettings(BaseSettings):
    """Generate settings. These are loaded from the .env config file. prefix=GS_

    Parameters
    ----------
    max_length : int
        Maximum length of the generated text
    num_beams : int
        Number of beams
    early_stopping : bool
        Whether to stop ealy if the beam search is not improving
    pad_token_id : int
        Token id for padding
    bos_token_id : int
        Token id for beginning of sentence
    no_repeat_ngram_size : int
        Maximum size of ngrams that can be repeated
    repetition_penalty : float
        Penalty for repetition
    """

    max_length: int = 2200
    num_beams: int = 30
    early_stopping: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    no_repeat_ngram_size: int = 2
    repetition_penalty: float = 0.001

    class Config:
        """Config"""

        env_prefix = "GS_"
        env_file = ".env"
        env_file_encoding = "utf-8"
