import pytest
import os
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def data_path() -> str:
    return os.path.dirname(os.path.realpath(__file__)) + "/data"

