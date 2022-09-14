"""
Transformer
"""

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def Transformer(vocab):
    """
    Creates a Transformer model with the given vocabulary size.

    Parameters
    ----------
    vocab:
        The vocabulary to use for the model.

    """
    ...
