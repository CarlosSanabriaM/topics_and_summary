#!/bin/bash

# Install dependencies using conda
conda install --file=/topics_and_summary/requirements.txt --channel conda-forge -y

# Install nltk resources
python -c "import nltk;nltk.download('stopwords');nltk.download('wordnet');nltk.download('punkt')"

# Install java
apt-get update -y && apt-get upgrade -y && apt-get install default-jdk -y

# Run the demo
python -c "import sys;sys.path.extend(['/topics_and_summary']);exec(open('/topics_and_summary/topics_and_summary/examples/demo.py').read())"
