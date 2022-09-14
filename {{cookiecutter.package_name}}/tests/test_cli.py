import os
import shutil
import pytest 

from {{cookiecutter.package_name}} import __version__ as VERSION
from {{cookiecutter.package_name}}.cli import cli


def test_{{cookiecutter.package_name}}(runner):
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["--version"])
        assert not result.exception
        assert result.output.split("\n")[-2] == f"{VERSION}"

@pytest.mark.skip(reason="Enable when you are ready")
def test_{{cookiecutter.package_name}}_preprocess(runner, data_path):
    with runner.isolated_filesystem():
        shutil.copytree(data_path, "data")
        shutil.copy(os.path.join(os.path.dirname(data_path), ".env"), ".env")
        result = runner.invoke(cli, ["--preprocess"])
        print(result.output)
        if not os.path.exists("data/cleandata"):
            raise Exception(f"data/cleandata does not exist. {result.exception}")
        if not os.path.exists("data/train.txt"):
            raise Exception(f"data/train.txt does not exist. {result.exception}")
        assert not result.exception


@pytest.mark.skip(reason="Enable when you are ready")
def test_{{cookiecutter.package_name}}_train(runner, data_path):
    with runner.isolated_filesystem():
        shutil.copytree(data_path, "data")
        shutil.copy(os.path.join(os.path.dirname(data_path), ".env"), ".env")
        result = runner.invoke(
            cli,
            [
                "--preprocess",
                "--train",
                "--generate",
            ],
        )
        print(result.output)
        if not os.path.exists("models/model.pt"):
            raise Exception(f"models/model.pt does not exist. {result.exception}")
        assert not result.exception

