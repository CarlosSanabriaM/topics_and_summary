.. _usage-installation:

Usage Installation
==================

.. note:: The instructions included here are only for **using** the library. Instructions to **contribute/modify** the library are
   present in the :ref:`development-installation` page.

To use this project, it have to be exported as library (python package).
For the moment, it isn't uploaded to any python packages repository, so the only way to obtain the project as a library
and install it is to download the source code and follow this steps:

1. Install the `dependencies`_ needed for using the library
2. `Install the library`_

Dependencies
------------

The packages for **using** the library are listed in the **requirements.txt** file, which content is showed below:

.. include:: ../../../../requirements.txt
   :literal:

They can be installed using `pip <https://pypi.org/project/pip/>`__ or
`conda <https://conda.io>`__.

The packages should be installed into an **enviroment**,
using either `virtualenv <https://virtualenv.pypa.io/en/latest/>`__
or `conda enviroments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__,
but conda is prefered.


Install dependencies using pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In a new virtualenv
"""""""""""""""""""
::

    # Create the new virtualenv
    virtualenv <env-name>

    # Activate the virtualenv
    source <env-name>/bin/activate
    # (<env-name>) should appear at the beginning of the prompt

    # Install the required packages to use the library
    pip install -r <path-to-project-root-folder>/requirements.txt
    # Packages will be installed in the <env-name> folder, avoiding conflicts with other projects


::

    # To leave the virtualenv run:
    deactivate
    # (<env-name>) should disappear form the beginning of the prompt

Install dependencies using conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In a new conda enviroment
"""""""""""""""""""""""""
::

    # Create the enviroment with the packages needed to use the library
    conda create --name=<enviroment-name> --file=<path-to-project-root-folder>/requirements.txt --channel conda-forge
    # Packages will be installed in the <enviroment-name> enviroment, avoiding conflicts with other enviroments

    # Change the current conda enviroment to the new enviroment
    conda activate <enviroment-name>
    # (<enviroment-name>) should appear at the beginning of the prompt, instead of (base)

::

    # To leave the conda enviroment run:
    conda deactivate
    # (base) should appear at the beginning of the prompt, instead of (<enviroment-name>)

In an existing conda enviroment
"""""""""""""""""""""""""""""""
::

    # Change the current conda enviroment to the existing enviroment
    conda activate <enviroment-name>
    # (<enviroment-name>) should appear at the beginning of the prompt, instead of (base)

    # Install the required packages to use the library
    conda install --file=<path-to-project-root-folder>/requirements.txt --channel conda-forge
    # Packages will be installed in the <enviroment-name> enviroment, avoiding conflicts with other enviroments

::

    # To leave the conda enviroment run:
    conda deactivate
    # (base) should appear at the beginning of the prompt, instead of (<enviroment-name>)


Install the library
-------------------
::

    pip install <path-to-project-root-folder>

To check if everything was installed correctly, execute the following commands:

::

    python
    >>> from topics_and_summary.datasets.twenty_news_groups import TwentyNewsGroupsDataset
    >>> dataset = TwentyNewsGroupsDataset()
    >>> dataset.print_some_files()
    # It should print some files of the dataset

.. warning:: After installing the library this way, **changes** in the source code **won't be reflected** in the library
    or the api documentation (even if generate-modules-doc.sh is executed, .rst files under the /api folder won't change).
    The :ref:`development-installation` page explains how to install the library in a way that this changes get reflected.