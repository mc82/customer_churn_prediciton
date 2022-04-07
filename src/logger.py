"""Set up the logging
__author__ = "Maik Goetze"
"""
import logging
from os import path

from costants import LOG_DIR
from directory import Directory

logging.basicConfig(
    filename=path.join(Directory(LOG_DIR).directory, 'churn_library.log'),
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)
