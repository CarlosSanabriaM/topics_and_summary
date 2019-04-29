#!/bin/sh

# Add the packages used in the project (only includes packages imported in .py files of external packages)
pipreqs topics_and_summary --force --savepath development-requirements.txt
# More info in https://github.com/bndr/pipreqs

# Change all the '_' in the name of the packages by '-', because packages are installed using '-' instead of '_'
# For example, 'scikit_learn' is isntalled as 'scikit-learn'
sed -i'.original' -e 's/_/-/g' development-requirements.txt
# Remove *.original files generated as backup by sed
rm *.original

# Add sphinx package needed for documentation
echo "sphinx==2.0.1" >> development-requirements.txt

# Add pipreqs package needed for generating the requirements.txt file (i'ts used in this file)
echo "pipreqs==0.4.9" >> development-requirements.txt
