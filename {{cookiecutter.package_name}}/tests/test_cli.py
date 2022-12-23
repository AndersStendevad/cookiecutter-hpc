import os
import shutil

from {{cookiecutter.package_name}} import __version__ as VERSION
from {{cookiecutter.package_name}}.cli import cli


def test_{{cookiecutter.package_name}}(runner):
    with runner.isolated_filesystem():
        os.mkdir("data")
        shutil.copytree(os.path.join(data_path, "raw_data"), "data/raw_data")
        result = runner.invoke(cli, ["--version"])
        assert not result.exception
        assert result.output.split("\n")[-2] == f"{VERSION}"


def test_{{cookiecutter.package_name}}_preprocess(runner, data_path):
    with runner.isolated_filesystem():
        os.mkdir("data")
        shutil.copytree(os.path.join(data_path, "raw_data"), "data/raw_data")
        shutil.copy(os.path.join(os.path.dirname(data_path), ".env"), ".env")
        result = runner.invoke(
            cli,
            [
                "--preprocess",
                "--data",
                data_path,
            ],
        )
        print(result.output)
        assert not result.exception


def test_{{cookiecutter.package_name}}_train_and_eval(runner, tmp_path_factory, data_path):
    fn = tmp_path_factory.mktemp("isolated_filesystem")
    with runner.isolated_filesystem(temp_dir=fn.absolute()):
        os.mkdir("data")
        shutil.copytree(os.path.join(data_path, "raw_data"), "data/raw_data")
        shutil.copy(os.path.join(os.path.dirname(data_path), ".env"), ".env")
        result = runner.invoke(
            cli,
            [
                "--run-id",
                "test",
                "--preprocess",
                "--train",
                "--eval",
                "--minutes",
                10,
                "--test-run",
            ],
        )
        if result.exception:
            print(result.output)
            print(result.exception)
            print(result.exception.__traceback__)
        assert result.exit_code == 0
        assert not result.exception
