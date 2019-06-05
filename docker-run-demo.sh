#!/bin/bash

# Run the demo
python -c "import sys;sys.path.extend(['/topics_and_summary']);exec(open('/topics_and_summary/topics_and_summary/examples/demo.py /topics_and_summary/datasets/20_newsgroups').read())"
