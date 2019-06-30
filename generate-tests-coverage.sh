#!/bin/sh

# Remove the previous .coverage file
coverage erase
# Generate the .coverage file with the tests coverage
# All the Python packages under the topics_and_summary folder whose test coverage want to be known must be added to the --source option, separated by commas
coverage run --branch --source=topics_and_summary/datasets,topics_and_summary/models,topics_and_summary/preprocessing topics_and_summary/tests/run_all_tests.py
# Generate the coverage.xml file using the .coverage file
coverage xml -i
