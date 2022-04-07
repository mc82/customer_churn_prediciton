"""Implements Directory class
__author__ = "Maik Goetze"
"""
from pathlib import Path

from logger import logging


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
        logging.info("Checking existence of directory {}".format(
            self._directory_name))
        exists = self._path.is_dir()
        logging.info("Directory exists: {}".format(exists))
        return exists

    def _create(self) -> None:
        """Creates directory
        """
        logging.info("Creating directory {}".format(self._directory_name))
        Path(self._path).mkdir(parents=True, exist_ok=True)
