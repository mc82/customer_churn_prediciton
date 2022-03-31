
"""Set up the logging
__author__ = "Maik Goetze"
"""
import logging


logging.basicConfig(
    filename='./output/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)
