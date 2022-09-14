"""Train
=====
Module for training loop of the model.
"""
import sys
from os.path import join
import os

import pendulum as pendulum
import pkbar
import time
import math
from {{cookiecutter.package_name}}.preprocess import (
    pad_EB_fixed,
    unsqueeze_list,
    remove_tracks,
    remove_wt_tokens,
    epoch_time,
    alt_eb_to_mid,
)
from {{cookiecutter.package_name}}.datasets import MusicDataset
from {{cookiecutter.package_name}}.model.trainer import Trainer
from {{cookiecutter.package_name}}.model.transformer import NystromTransformer, device
from {{cookiecutter.package_name}}.model.vocab import Vocab
from {{cookiecutter.package_name}}.model.settings import TrainerSettings, GenerateSettings

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader


def train(
    data_path: str,
    model_path: str,
    hours: int = 60,
    minutes: int = 0,
    logfile: str = "log.txt",
):
    """
    Train loop for the Transformer model.

    Parameters
    ----------
    data_path: str
        Path to the dataset folder.
    model_path: str
        Path to the model.
    hours: int = 60
        Number of hours to train for.
    minutes: int = 0
        Number of minutes to train for.
    logfile: str = "log.txt"
        Path to the log file.
    """

    settings = TrainerSettings()

    batch_size = settings.batch_size
    lr = settings.lr

    vocab = Vocab()
    note_to_idx = vocab.note_to_idx

    print("Composing model")
    tsfrm = transforms.Compose(
        [
            pad_EB_fixed(N=200),
            unsqueeze_list(),
            remove_tracks(),
            remove_wt_tokens(),
        ]
    )

    print("Loading data")
    train_dataset = MusicDataset(join(data_path, "train.txt"), transform=tsfrm)
    valid_dataset = MusicDataset(join(data_path, "val.txt"), transform=tsfrm)
    X_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vocab.generate_batch,
    )
    X_valid = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vocab.generate_batch,
    )
    designated_sample = next(X_valid.__iter__())[0].unsqueeze(0).to(device)

    print(f"Setting up model and sending to {device} device")
    model = NystromTransformer(vocab).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=note_to_idx["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, optimizer, criterion, 1, device)

    now = pendulum.now()
    to_time = now.add(hours=hours, minutes=minutes)
    print(f"Start training with limit of {to_time.diff(now).in_words()}")

    with open(logfile, "w") as f:
        f.write("time,train_loss,train_ppl,val_loss,val_ppl\n")

    for epoch in range(sys.maxsize):
        if pendulum.now() >= to_time:
            break

        model.train()

        progress = (
            pkbar.Kbar(target=len(X_train), width=25) if device == "cpu" else None
        )

        start_time = time.time()
        train_loss = trainer.train(X_train, progress)
        valid_loss = trainer.eval(X_valid)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print()
        print(f"Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
        )
        print(
            f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}"
        )
        print(f"Time left of training: {to_time.diff(pendulum.now()).in_words()}")
        print()

        torch.save(model.state_dict(), model_path)

        if epoch % 10 == 0:
            filepath = os.path.join(
                os.path.dirname(model_path),
                "snapshots",
                f"model_epoch_{epoch}.pt",
            )
            print(f"Saving snapshot at {filepath}")
            torch.save(model.state_dict(), filepath)
            model.eval()

            sample_filepath = os.path.join(
                os.path.dirname(model_path),
                "snapshots",
                f"generated_sample_epoch_{epoch}.mid",
            )

            print("Generating sample.")
            preds = model.generate(
                designated_sample[:, : min(designated_sample.shape[1], 200)],
                **GenerateSettings().dict(),
            ).squeeze()

            out = []
            for t in preds[min(designated_sample.shape[1], 200) :]:  # noqa
                out.append(vocab.idx_to_note[int(t)])

            if out:

                with open(
                    os.path.join(
                        os.path.dirname(model_path),
                        "snapshots",
                        f"generated_sample_epoch_{epoch}.csv",
                    ),
                    "w",
                ) as f:
                    f.write(",".join(out))

                mid = alt_eb_to_mid(out.copy())
                mid.save(sample_filepath)
                print(f"Saved sample to {sample_filepath}")
            else:
                print("No sample generated.")

        with open(logfile, "a") as f:
            f.write(
                f"{pendulum.now()},{train_loss},{math.exp(train_loss)},{valid_loss},{math.exp(valid_loss)}\n"
            )
