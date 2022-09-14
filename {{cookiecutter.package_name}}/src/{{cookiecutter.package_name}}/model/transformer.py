"""
Transformer
"""

from transformers import NystromformerForMaskedLM, NystromformerConfig

from {{cookiecutter.package_name}}.model.settings import ModelSettings

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def NystromTransformer(vocab):
    """
    Creates a Nyströmformer model with the given vocabulary size.

    Parameters
    ----------
    vocab : Vocab
        The vocabulary to use for the model.

    Returns
    -------
    model : NystromformerForMaskedLM
        The Nyströmformer model.
    """

    config = NystromformerConfig(
        vocab_size=len(vocab),
        **ModelSettings().dict(),
    )
    return NystromformerForMaskedLM(config=config)
