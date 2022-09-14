"""{{cookiecutter.package_name}} CLI"""
import os
import random
import transformers

import click
import {{cookiecutter.package_name}}
from {{cookiecutter.package_name}} import __version__ as VERSION
from {{cookiecutter.package_name}}.preprocess import (
    DataStreamer,
    split,
)


@click.command()
@click.option("--version", is_flag=True, help="Shows the version of {{cookiecutter.package_name}}")
@click.option("--seed", type=int, help="Set a seed")
@click.option(
    "--data",
    type=click.Path(),
    default="data/",
    help="Optional datafolder where data is when downloading, preprocessing and using. Defaults to ./data .",
)
@click.option(
    "--model",
    type=click.Path(),
    default="models/model.pt",
    help="Optional model path where model is saved to and loaded from. Defaults to model/model.pt",
)
@click.option(
    "--out",
    type=click.Path(),
    default="out/",
    help="Optional outfolder where output is saved to. Defaults to out/.",
)
@click.option("--download", is_flag=True, help="Downloads the dataset")
@click.option("--preprocess", is_flag=True, help="Prepares the dataset")
@click.option(
    "--preprocess-pack",
    type=click.Choice(["standard"], case_sensitive=False),
    default="standard",
    help="Preprocesses the dataset with the specified pack. Defaults to 'standard'.",
)
@click.option("--train", is_flag=True, help="Train a model.")
@click.option(
    "--hours",
    type=int,
    default=0,
    help="Number of hours to train for. Sum of hours and minutes is total "
    "duration. Default is 1 minutes.",
)
@click.option(
    "--minutes",
    type=int,
    default=1,
    help="Number of minutes to train for. Sum of hours and minutes is "
    "total duration. Default is 1 minutes.",
)
@click.option(
    "--logfile",
    type=click.Path(),
    default="log.txt",
    help="Logfile to save to. Defaults to log.txt",
)
@click.option("--generate", is_flag=True, help="Generate Music")
@click.option(
    "--max-gen",
    type=int,
    default=1000,
    help="Maximum number of samples to generate from test set. " "Defaults to 1000.",
)
def cli(
    version,
    seed,
    data,
    model,
    out,
    download,
    preprocess,
    preprocess_pack,
    train,
    hours,
    minutes,
    logfile,
    generate,
    max_gen,
):
    """The main way to engage with {{cookiecutter.package_name}} is with this cli"""

    if not seed:
        seed = random.randint(0, 2**32 - 1)

    print(f"Running with seed {seed}")
    transformers.set_seed(seed)

    if version:
        click.echo(f"{VERSION}")

    if download or preprocess:
        if not os.path.exists(data):
            print(f"Creating directory {data} because it does not exists")
            os.mkdir(data)

    if download:
        raise Exception(
            f"""Download not implemented. Get the data yourself and place it in {os.path.join(data, "rawdata")}"""
        )

    if preprocess:

        datastreamer = DataStreamer(data)

        if preprocess_pack == "standard":
            transforms = []
        else:
            transforms = []

        print(
            f"""Streaming data. From {os.path.join(data, "rawdata")} to {os.path.join(data, "predata")}"""  # noqa
        )
        datastreamer.preprocess(transforms)

        print(
            f"""Streaming data. From {os.path.join(data, "predata")} to {os.path.join(data, "cleandata")}"""  # noqa
        )
        datastreamer.preprocess([])
        split(data_path=data)

    if train:
        directory = os.path.dirname(model)
        if not os.path.exists(directory):
            print(f"Creating directory {directory} because it does not exists")
            os.mkdir(directory)
        if not os.path.exists(os.path.join(directory, "snapshots")):
            print(
                f"Creating directory {os.path.join(directory, 'snapshots')} because it does not exists"
            )
            os.mkdir(os.path.join(directory, "snapshots"))

        print("Training model...")
        {{cookiecutter.package_name}}.model.train(
            data_path=data,
            model_path=model,
            hours=hours,
            minutes=minutes,
            logfile=logfile,
        )

    if generate:
        if not os.path.exists(model):
            raise Exception(f"Model is not trained yet. No model found at {model}")
        if not os.path.exists(out):
            print(f"Creating directory {out} because it does not exists")
            os.mkdir(out)
        {{cookiecutter.package_name}}.model.generate(
            data_path=data, model_path=model, output_path=out, max_gen=max_gen
        )
