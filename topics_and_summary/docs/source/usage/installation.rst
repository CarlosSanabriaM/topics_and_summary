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


.. _usage-installation-dependencies:





Dependencies
------------

The packages for **using** the library are listed in the **requirements.txt** file, which content is showed below:

.. include:: ../../../../requirements.txt
   :literal:

They can be installed using `pip <https://pypi.org/project/pip/>`__ or
`conda <https://conda.io>`__.

The packages should be installed into an **environment**,
using either `virtualenv <https://virtualenv.pypa.io/en/latest/>`__
or `conda environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__,
but conda is preferred.


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
In a new conda environment
""""""""""""""""""""""""""
::

    # Create the environment with the packages needed to use the library
    conda create --name=<environment-name> --file=<path-to-project-root-folder>/requirements.txt --channel conda-forge
    # Packages will be installed in the <environment-name> environment, avoiding conflicts with other environments

    # Change the current conda environment to the new environment
    conda activate <environment-name>
    # (<environment-name>) should appear at the beginning of the prompt, instead of (base)

::

    # To leave the conda environment run:
    conda deactivate
    # (base) should appear at the beginning of the prompt, instead of (<environment-name>)

In an existing conda environment
""""""""""""""""""""""""""""""""
::

    # Change the current conda environment to the existing environment
    conda activate <environment-name>
    # (<environment-name>) should appear at the beginning of the prompt, instead of (base)

    # Install the required packages to use the library
    conda install --file=<path-to-project-root-folder>/requirements.txt --channel conda-forge
    # Packages will be installed in the <environment-name> environment, avoiding conflicts with other environments

::

    # To leave the conda environment run:
    conda deactivate
    # (base) should appear at the beginning of the prompt, instead of (<environment-name>)


.. _usage-installation-library:





Install the library
-------------------

Execute the following command inside the venv or conda environment:

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

.. _usage-installation-other-elements:





Donwload NLTK resources
-----------------------

The following NLTK resources are required by the preprocessing package:

* stopwords
* wordnet

To install them, follow this steps from the command line, inside the venv or conda environment:

::

    python
    >>> import nltk
    >>> nltk.download('stopwords')
    # It should print True
    >>> nltk.download('wordnet')
    # It should print True
    >>> nltk.download('punkt')
    # It should print True



.. _download-other-elements:

Download other elements
-----------------------

This section explains how to install other elements that may be required for certain funcionality:

.. _usage-installation-mallet:

Download mallet
^^^^^^^^^^^^^^^

`Mallet <http://mallet.cs.umass.edu>`__ is a Java library for NLP, and is **required to use the LdaMalletModel**.

It's written in Java, so **it requires that Java is installed**.

Download mallet library
"""""""""""""""""""""""

Mallet source code can be downloaded from the `mallet download page <http://mallet.cs.umass.edu/download.php>`__.

`This is a direct link for downloading the mallet 2.0.8 version in .tar.gz format. <http://mallet.cs.umass.edu/dist/mallet-2.0.8.tar.gz>`__

Download Java
"""""""""""""

Mallet requires the Java JDK 1.8.

In ubuntu, it can be installed executing the following command:

::

    sudo apt-get install default-jdk

For instructions on how to install Java JDK 1.8 in other systems see the `Open JDK page <https://openjdk.java.net/install/>`__
and the `Oracle JDK 1.8 page <https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html>`__.


Download word embeddings
^^^^^^^^^^^^^^^^^^^^^^^^

**At least one** of the following word embeddings **is required to use the TextRank** algorithm for text summarization.

Download  Word2Vec
""""""""""""""""""

Word2Vec pretrained vectors can be downloaded from the `Google Code Archive Word2Vec page <https://code.google.com/archive/p/word2vec/>`__.

`This is a direct link for downloading the 300-dimensional word vectors pretrained on Google News dataset in .bin.gz format.
<https://drive.google.com/uc?export=download&confirm=MN4d&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM>`__

.. warning:: After clicking the above link, the download will start, and the size of the .bin.gz file is **1.65GB**.


Download  GloVe
"""""""""""""""

GloVe pretrained vectors can be downloaded from the `Stanford NLP GloVe page <https://nlp.stanford.edu/projects/glove/>`__.

`This is a direct link for downloading the 50, 100, 200 and 300-dimensional word vectors pretrained on Wikipedia in .zip format.
<http://nlp.stanford.edu/data/glove.6B.zip>`__

.. warning:: After clicking the above link, the download will start, and the size of the .zip file is **862MB**.

Download  20 NewsGroups dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The 20 newsgroups dataset can be used to try the functionality of this library. It can be downloaded from the
`UCI Donald Bren School of Information & Computer Sciences 20 newsgroups page
<https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/>`__.

`This is a direct link for downloading the 20 newsgroups dataset in .tar.gz format.
<https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz>`__

.. warning:: After clicking the above link, the download will start, and the size of the .tar.gz file is **17.3MB**.
