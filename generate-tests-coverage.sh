#!/bin/sh

coverage erase
coverage run --branch --source=topics_and_summary/datasets,topics_and_summary/models,topics_and_summary/preprocessing topics_and_summary/tests/run_all_tests.py
coverage xml -i
