"""Implements tests of Directory class
__author__ = "Maik Goetze"
"""
from os import path
from os import makedirs
import pathlib
import pytest
from directory import Directory


@pytest.fixture
def directory_exists(tmpdir):
    makedirs(path.join(tmpdir, "test"))
    directory = Directory(path.join(tmpdir, "test"))
    return directory


@pytest.fixture
def directory_not_exists(tmpdir):
    test_dir = "test"
    directory = Directory(path.join(tmpdir, test_dir))
    return directory


def test_path_exists(directory_exists):
    assert isinstance(directory_exists.directory, pathlib.Path)


def test_path_not_exists(directory_not_exists):
    assert isinstance(directory_not_exists.directory, pathlib.Path)
