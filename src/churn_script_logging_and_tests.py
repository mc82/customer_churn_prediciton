from logger import logging
import pytest

if __name__ == '__main__':
    logging.info("Running tests with pytest....")
    pytest.main(["--report-log", "output/test.log"])
    logging.info("Running tests completed")
