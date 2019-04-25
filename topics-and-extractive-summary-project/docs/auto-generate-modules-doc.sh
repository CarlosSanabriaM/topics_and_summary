#!/bin/sh
cd source/

# Add here the packages names.
#rm embeddings.rst
#rm utils.rst
#rm visualizations.rst
#rm datasets.rst
#rm examples.rst
#rm models.rst
#rm preprocessing.rst
#rm tests.rst
#rm modules.rst

# Generates the automatic documentation of the modules in the project
# See https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html for sphinx-apidoc options
sphinx-apidoc -o api ../.. --force --separate --module-first

# Generates the html files in the build folder
cd ..
make html
