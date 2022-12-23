"""{{cookiecutter.package_name}} CLI"""
import os
import random
import transformers
import click
import uuid

from os.path import abspath
from {{cookiecutter.package_name}} import __version__ as VERSION
from {{cookiecutter.package_name}}.preprocess import (
    DataStreamer,
    NoiseStreamer,
)
from rifs.model import fine_tune, evaluate


@click.command()
@click.option("--version", is_flag=True, help="Shows the version of rifs")
@click.option("--seed", type=int, help="Set a seed")
@click.option("--run-id", type=str, help="Set a run-id")
@click.option(
    "--data",
    type=click.Path(exists=False),
    default="",
    help="Optional datafolder for where all data is placed and used from. Defaults to ./data .",
)
@click.option(
    "--model",
    type=click.Path(),
    default="huggingface_models",
    help="Optional model path where model is saved to and loaded from. Defaults to huggingface_models/ ."
    " Actual model will be under a subfolder with the run-id.",
)
@click.option(
    "--pretrained",
    type=str,
    default="Alvenir/wav2vec2-base-da",
    help="Optional pretrained model name. Defaults to Alvenir/wav2vec2-base-da ."
    " See https://huggingface.co/models for more models.",
)
@click.option("--download", is_flag=True, help="Downloads the dataset")
@click.option("--preprocess", is_flag=True, help="Prepares the dataset")
@click.option("--train", is_flag=True, help="Train a model.")
@click.option("--eval", is_flag=True, help="Evaluate test set")
@click.option(
    "--output",
    type=str,
    default="output",
    help="Output folder for evaluate. Defaults to ./output",
)
@click.option(
    "--eval-snapshots",
    is_flag=True,
    help="Also evaluate snapshot models saved at steps.",
)
@click.option(
    "--eval-gold",
    is_flag=True,
    help="Evaluate gold test set. Defaults to False. Otherwise uses validation set.",
)
@click.option(
    "--max-predict",
    type=int,
    default=100,
    help="Max number of predictions to output. Default is 100",
)
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
@click.option("--test-run", is_flag=True, help="Runs model with reduced parameters")
@click.option(
    "--warmup-steps",
    type=int,
    default=0,
    help="Warmup steps for fine_tune, Default is 0",
)
def cli(
    version,
    seed,
    run_id,
    data,
    model,
    pretrained,
    download,
    preprocess,
    train,
    eval,
    output,
    eval_snapshots,
    eval_gold,
    max_predict,
    hours,
    minutes,
    test_run,
    warmup_steps,
):
    """The main way to engage with {{cookiecutter.package_name}} is with this cli"""
    if version:
        click.echo("{{cookiecutter.package_name}} version", VERSION)
        return
    if not data:
        data = abspath("data")

        return
    if preprocess or train or eval:
        if not os.path.exists(data):
            click.echo(
                f"Data folder {data} does not exist. Please download data before preprocessing, training or evaluating.", err=True
            )
            return
    if not seed:
        seed = random.randint(0, 2**32 - 1)

    click.echo(f"Running with seed {seed}")
    transformers.set_seed(seed)

    if not run_id:
        run_id = str(uuid.uuid4())
    click.echo(f"Running with run_id {run_id}")

    if version:
        click.echo(f"{VERSION}")

    if download or preprocess:
        if not os.path.exists(data):
            click.echo(f"Creating directory {data} because it does not exists")
            os.mkdir(data)

    if download:
        click.echo(
            f"""Download not implemented. Get the data yourself and place it in {os.path.join(data, "raw_data")}""", err=True
        )

    if preprocess:
        transforms = []
        datastreamer = DataStreamer(data)

        click.echo(
            f"""Streaming data. From {os.path.join(data, "raw_data")} to {os.path.join(data, "clean_data")}"""  # noqa
        )
        datastreamer.preprocess(transforms)

    final_model_path = os.path.join(model, run_id)

    if train:
        if not os.path.exists(model):
            click.echo(f"Creating directory {model} because it does not exist")
            os.mkdir(model)
        train_file = (os.path.join(data, "raw_data", "train")
        test_file = (os.path.join(data, "raw_data", "dev")


        click.echo("Training model...")
        fine_tune(
            train_file=train_file,
            test_file=test_file,
            data_path=data,
            model_path=final_model_path,
            from_pretrained_name=pretrained,
            run_id=run_id,
            hours=hours,
            minutes=minutes,
            test=test_run,
            warmup_steps=warmup_steps,
        )

    if eval:
        if not os.path.exists(os.path.join(model, run_id)):
            click.echo((
                f"Model is not trained yet. No model found at {final_model_path}"
            , err=True))
            return
        click.echo("Evaluating model...")
        test_file = (
            os.path.join(data, "raw_data", "dev")
            if not eval_gold
            else os.path.join(data, "raw_data", "test")
        )
        evaluate(
            test_file=test_file,
            data_path=data,
            model_path=final_model_path,
            output_path=output,
            noise_pack=noise_pack,
            run_id=run_id,
            eval_snapshots=eval_snapshots,
            max_predict=max_predict,
        )

