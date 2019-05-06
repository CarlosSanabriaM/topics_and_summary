.. _development-installation:

Development Installation
========================

.. note:: The instructions included here are for **contributing** to the library. Instructions to **use** the library are
   present in the :ref:`usage-installation` page.

The installation steps are very similar as the ones described in the :ref:`usage-installation` page. The main differences are:

* The requested **dependencies**
* The **way the library is installed** with pip


.. _development-installation-dependencies:

Dependencies
------------

The packages for **modifying** the library are listed in the **development-requirements.txt** file, which content is showed below:

.. include:: ../../../../development-requirements.txt
   :literal:

Basically, there are 2 additional packages required:

* **sphinx:** For generating the **documentation**.
* **pipreqs:** For generating the **requirements.txt** and **development-requirements.txt** files.

The way this dependencies are installed is the same as the one described in the :ref:`usage-installation-dependencies`
section of the Usage Installation page, but **changing the requirements.txt file by development-requirements.txt.**


.. _development-installation-install-the-library:

Install the library
-------------------

The way of installing the library is also similar as the one described in the :ref:`usage-installation-library`
section of the Usage Installation page, but in this case, the *-e* option of the *pip install* command is used.
This option is described in the
`Python Packaging User Guide <https://packaging.python.org/tutorials/installing-packages/#installing-from-a-local-src-tree>`__.

Basically, after installing the package this way, **all changes made in the source code will be reflected in the installed library.**
This is because it's a "fake installation". It doesn't copy any files. Rather, it points to the source code folder.
This allows to have an "always updated" version of the library installed.

The **main objective** of installing the library this way is to allow sphinx to **generate the documentation of the last version** of the source code.
The **generate-modules-doc.sh file** generates the documentation of the modules of the library **using the content of the installed library** [#f1]_,
so this way of installing it allows to keep the same content in the source code and in the installed library.

::

    pip install -e <path-to-project-root-folder>

Install other elements
----------------------

The installation steps are the same as the ones described in the :ref:`usage-installation-other-elements`
section of the Usage Installation page.


.. rubric:: Footnotes

.. [#f1] The documentation can't be generated directly using the content of the modules in the source code, because
    the source code includes the name of the library in all the imports that refer to modules of the own library [#f2]_
    (this is needed to allow the project to be converted into an installable library without problems in the imports),
    and this causes problems while trying to generate the documentation using the source code. So the solution is to
    install the library and obtain the code from that installed library.

.. [#f2] For example: from **topics_and_summary**.datasets.common import get_file_content, Document, Dataset