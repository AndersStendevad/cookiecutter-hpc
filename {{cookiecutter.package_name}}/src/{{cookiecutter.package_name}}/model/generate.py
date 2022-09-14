"""Generate
========

Module for generating the model.
"""

import os.path
from os.path import join

from {{cookiecutter.package_name}}.datasets import MusicDataset
from {{cookiecutter.package_name}}.model.transformer import NystromTransformer, device
from {{cookiecutter.package_name}}.model.vocab import Vocab
from {{cookiecutter.package_name}}.model.settings import GenerateSettings

from torch.utils.data import DataLoader

import torch


def generate(data_path: str, model_path: str, output_path: str, max_gen: int = -1):
    """
    Generate music from a trained model.

    Parameters
    ----------
    data_path : str
        Path to the dataset.

    model_path : str
        Path to the model.

    output_path : str
        Path to the output file.

    max_gen : int
        Maximum number of generated files. Default: -1 (all files).
    """

    vocab = Vocab()

    test_dataset = MusicDataset(join(data_path, "test.txt"))
    X_test = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=vocab.generate_batch,
    )

    model = NystromTransformer(vocab).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for i, src in enumerate(X_test):

        seq = torch.squeeze(src[:, :, : min(200, src.shape[1])], dim=0).to(device)
        preds = model.generate(seq, **GenerateSettings().dict()).squeeze()

        out = []
        for t in preds[min(seq.shape[1], 200) :]:  # noqa
            out.append(vocab.idx_to_note[int(t)])

        with open(os.path.join(output_path, f"generated_sample_{i}.csv"), "w") as f:
            f.write(",".join(out))

        mid = alt_eb_to_mid(out.copy())
        mid.save(os.path.join(output_path, f"generated_sample_{i}.mid"))

        print(f"Generated sample number {i} to {output_path}")

        if i == max_gen:
            break
