import logging

import Htool


def test_logger():
    logging.basicConfig(level=logging.DEBUG)
    Htool.test_logger()
