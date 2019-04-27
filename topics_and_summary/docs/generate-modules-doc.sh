#!/bin/sh
cd source/

# Generates the automatic documentation of the modules in the project
# See https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html for sphinx-apidoc options
# The paths that follows 'phinx-apidoc -o api ../..' are paths excluded from the generation
sphinx-apidoc -o api ../.. ../../examples ../../tests --force --separate --module-first

# Generates the html files in the build folder
cd ..
make html
