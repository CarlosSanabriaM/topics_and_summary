.. _development-api:

Development: API
================

This page explains how to **modify the API**. In specific:

* Explains the bases of the library (AI algorithms used, important libraries, main concepts, ...)
* Gives some tips on how to extend it's functionality.


.. warning:: All the **imports** that refer to modules of the library must specifiy the **name of the library**:
    ::

        from topics_and_summary.datasets.common import get_file_content, Document, Dataset

Dependencies
------------

.. I don't know why :ref:`development-installation` doens't work as it does in usage/installation.rst

The instructions for installing the required dependencies are included in the :doc:`installation` page.

Code style
----------

The code style must follow the :pep:`8` style guide and the **tab size** must be of **4 spaces**.

Main concepts
-------------



Extending the library
---------------------

Adding new datasets
^^^^^^^^^^^^^^^^^^^
The steps to add a new class are the following:

1. Create a new folder with the dataset inside the **datasets folder**
2.


