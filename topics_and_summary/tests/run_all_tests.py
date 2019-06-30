"""
Module to run all tests.

This module should only be used by the Coverage.py module to generate tests coverage and tests reports,
because the results this module gives to the user about the tests execution are worst than the ones given by the PyCharm IDE.

In PyCharm, to execute all tests and see the results properly, right click the topics_and_summary/tests folder
and select "Run 'unittests in tests'" option.
"""

import unittest

from topics_and_summary.tests.paths import TESTS_BASE_PATH

if __name__ == '__main__':
    loader = unittest.TestLoader()

    # Obtain a suit of tests with all the tests in the tests directory
    suite = loader.discover(TESTS_BASE_PATH)

    # Run all tests
    runner = unittest.TextTestRunner()
    runner.run(suite)
