"""Implements Directory class
__author__ = "Maik Goetze"
"""
from pathlib import Path


class Directory:
    """Implements logic to create a directory if not exists and returns the path to it
    """

    def __init__(self, directory_name: str) -> None:
        self._directory_name = directory_name
        self._path = Path(self._directory_name)

    @property
    def directory(self) -> Path:
        """_summary_

        :return: Path of the directory
        :rtype: Path
        """
        if not self._check_existence():
            self._create()
        return self._path

    def _check_existence(self) -> bool:
        """Checks existens of directory
        :return: returns True if directory exists otherwise False
        :rtype: bool
        """
        exists = self._path.is_dir()
        return exists

    def _create(self) -> None:
        """Creates directory
        """
        Path(self._path).mkdir(parents=True, exist_ok=True)
