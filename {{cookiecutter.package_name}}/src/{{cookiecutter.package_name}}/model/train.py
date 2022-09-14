"""Train
=====
Module for training loop of the model.
"""
import sys
import os

import pendulum as pendulum
import pkbar
import time
import math

from {{cookiecutter.package_name}}.model.transformer import device


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

    print(data_path)

    print("Composing model")

    print("Loading data")

    print(f"Setting up model and sending to {device} device")

    now = pendulum.now()
    to_time = now.add(hours=hours, minutes=minutes)
    print(f"Start training with limit of {to_time.diff(now).in_words()}")

    with open(logfile, "w") as f:
        f.write("time,train_loss,train_ppl,val_loss,val_ppl\n")

    for epoch in range(sys.maxsize):
        if pendulum.now() >= to_time:
            break

        X_train = [1]
        progress = (
            pkbar.Kbar(target=len(X_train), width=25) if device == "cpu" else None
        )

        start_time = time.time()

        print(progress)

        train_loss = 0
        valid_loss = 0
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

        if epoch % 10 == 0:
            filepath = os.path.join(
                os.path.dirname(model_path),
                "snapshots",
                f"model_epoch_{epoch}.pt",
            )
            print(f"Saving snapshot at {filepath}")

            sample_filepath = os.path.join(
                os.path.dirname(model_path),
                "snapshots",
                f"generated_sample_epoch_{epoch}.mid",
            )

            print("Generating sample.")

            print(f"Saved sample to {sample_filepath}")

        with open(logfile, "a") as f:
            f.write(
                f"{pendulum.now()},{train_loss},{math.exp(train_loss)},{valid_loss},{math.exp(valid_loss)}\n"
            )


def epoch_time(start_time: int, end_time: int):
    """
    Calculates the time it took to complete an epoch.
    Parameters
    ----------
    start_time
    end_time

    Returns
    -------

    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
