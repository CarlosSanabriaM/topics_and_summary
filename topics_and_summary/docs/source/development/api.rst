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

Topics
^^^^^^
The topics are obtained using either LSA or LDA algorithms. To obtain the full functionality, LDA must be used, because
LSA doesn't give probability values to the words inside a topics (in LSA the value of a word inside a topic can be negative!),
and most of the functionality is based on probabilities.

On the other hand, LDA shows way better results than LSA, at least in practice. Also, LSA throws an exception when triying
to plot the wordclouds of it's topics, probably because some words have negative values, as explained above.

The only reason why LSA is kept in this library is because it was the first algorithm tried, and it can be used
to compare it's results against LDA.


Below are some links to papers and posts about both algorithms:

LSA
"""
**Papers:**

* ads
* as

**Posts:**

* ads
* as

LDA
"""
asd



Summary
^^^^^^^
sdff


Extending the library
---------------------

Adding new datasets
^^^^^^^^^^^^^^^^^^^
The steps to add a new class are the following:

1. Create a new folder with the dataset inside the **datasets folder**
2.


