#!/bin/sh

# Add the packages used in the project (only includes packages imported in .py files of external packages)
pipreqs topics_and_summary --force --savepath requiremens.txt
# More info in https://github.com/bndr/pipreqs

# Add sphinx package needed for documentation
echo "sphinx==2.0.1" >> file

# Add pipreqs package needed for generating the requiremens.txt file (i'ts used in this file)
echo "pipreqs==0.4.9" >> file
