.. _development-api:

Development: API
================

This page explains:

* The `code style`_ used in the library
* The `main concepts`_ of the topics extraction and the summarization models
* The most `important external libraries`_ used in the source code
* The `directory structure and important files`_
* `Important source code details`_
* Tips to `extend the library`_

.. warning:: All the **imports** that refer to modules of the library must specifiy the **name of the library**:
    ::

        from topics_and_summary.datasets.common import get_file_content, Document, Dataset





Dependencies
------------

.. I don't know why :ref:`development-installation` doens't work as it does in usage/installation.rst

The instructions for installing the required dependencies are included in the
:ref:`development-installation-dependencies` section of the Development Installation page.





Code style
----------

The code style must follow the :pep:`8` style guide and the **tab size** must be of **4 spaces**.





Main concepts
-------------

Topics
^^^^^^

The topics are obtained using either **LSA** or **LDA** algorithms. LSA means Latent Semantic Analysis (also known as Latent
Semantic Indexing), and LDA means Latent Dirichlet Allocation.

In this library, 2 different implementations of the LDA algorithm are used:

* Gensim LDA implementation
* Gensim Wrapper of the Mallet LDA implementation

`Mallet <http://mallet.cs.umass.edu>`__ is a Java library for NLP. The **LDA mallet implementation shows way better results
than the LDA gensim native implementation**.

To obtain the full functionality of this library, the **LDA algorithm must be used**, instead of using LSA, because LSA
doesn't give probability values to the words inside a topic (in LSA the value of a word inside a topic can be negative!),
and most of the functionality of the library is based on those probabilities, so LSA doesn't allow to use all the functionality.

Moreover, **LDA shows way better results than LSA**, at least in practice. Another problem of LSA is that it throws
an exception when trying to plot the wordclouds of it's topics, probably because some words have negative values, as explained above.

The only reason why LSA is kept in this library is because it was the first algorithm tried, and it can be used
to compare it's results against LDA.

Below are some links to information about both algorithms:

LSA
"""

* `LSA (DataCamp) <https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python>`__
* `LSA implementation in gensim <https://radimrehurek.com/gensim/models/lsimodel.html>`__

LDA
"""

* `LDA implementation in gensim <https://radimrehurek.com/gensim/models/ldamodel.html>`__
* `LDA mallet implementation in gensim <https://radimrehurek.com/gensim/models/wrappers/ldamallet.html>`__

Topic Modelling algorithms overview
"""""""""""""""""""""""""""""""""""

* `LSA, LDA, LDA2VEC and PLSA overview <https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05>`__

Summarization
^^^^^^^^^^^^^

For text summarization, **the algorithm implemented in this library is TextRank**, but there are other ways of generating
text summaries, i.e. with Deep Learning.

The following paper reviews the **main approaches to automatic text summarization**:
`Text Summarization Techniques: A Brief Survey <https://arxiv.org/pdf/1707.02268.pdf>`__.

Below are some links to information about TextRank and other alternatives for generating the summaries:

TextRank
""""""""

* `Text Rank in Python <https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/>`__

Other summarization alternatives
""""""""""""""""""""""""""""""""

* `Summarization with LDA <https://dzone.com/articles/lda-for-text-summarization-and-topic-detection>`__
* `Summarization with Deep Learning <https://hackernoon.com/text-summarizer-using-deep-learning-made-easy-490880df6cd>`__
* `Summarization with Keras library <https://hackernoon.com/text-summarization-using-keras-models-366b002408d9>`__





Important external libraries
----------------------------

Gensim
^^^^^^

This library is the most important one, because the topics are obtained using it's LSA and LDA implementations.

`This is a direct link to the gensim documentation. <https://radimrehurek.com/gensim/>`__

Pandas
^^^^^^

This library is also vey important, because a lot of the functionality of this library uses pandas DataFrames.

`This is a direct link to the pandas documentation. <http://pandas.pydata.org/pandas-docs/stable/>`__





Directory structure and important files
---------------------------------------

* **datasets** folder: Here is where the dataset have to be stored, each dataset in it's own folder.
* **embeddings** folder: Here is where the pretrained word embeddings have to be stored. It must have the following folders:

   * glove: Contains the glove.6B folder, obtained after decompressing the glove.6B.zip file.
   * word2vec: Contains the GoogleNews-vectors-negative300.bin.gz file.

* **mallet-2.0.8** folder: Is the mallet source code folder, obtained after decompressing the mallet-2.0.8.tar.gz file.
* **topics_and_summary** folder: Is the python package of the library. It contains all the source code of the library,
   and has the following elements:

   * **datasets** folder: Python package with functionality for the datasets.
   * **docs** folder: Contains all the documentation files.
   * **examples** folder: Python package with some examples and utilities for generating models.
   * **logs** folder: Contains the logs.
   * **models** folder: Python package with the topics and summary modules. **This is the most important package.**
   * **notebooks** folder: Contains Jupyter Notebooks with some tests.
   * **preprocessing** folder: Python package with all the preprocessing functionality.
   * **preprocessing-files** folder: Contains files used in the preprocessing.text.py module.
   * **saved-elements** folder: Contains functions, objects and models stored on disk.
   * **tests** folder: Python package with all the unit tests.
   * **embeddings** folder: Python module with the Glove and the Word2Vec classes.
   * **utils** folder: Python module with some utilities: paths and save/load objects and functions.
   * **visualizations** folder: Python module that plots the wordclouds and generates the tsne visualization of the topics.

* **generate-development-requirements.sh**: Creates the development-requirements.txt file based on the libraries used in the source code.
* **generate-usage-requirements.sh**: Creates the requirements.txt file based on the libraries used in the source code.
* **setup.py**: File used to install the library. It contains information about the library installation.
* **sonar-project.properties**: Contains the configuration for the SonarQube static code analysis tool.


.. note:: The instructions to obtain the glove.6B.zip, the GoogleNews-vectors-negative300.bin.gz and the mallet-2.0.8.tar.gz
   files are described in the :ref:`usage-installation-other-elements` section of the Usage Installation page.





Important source code details
-----------------------------

predict_topic_prob_on_text()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The method **predict_topic_prob_on_text()** of the models.topics.TopicsModel class is the one that predicts the topics probability
of a given text. With LsaModel and LdaModel this method takes less than a second. But, **with LdaMalletModel, this methods
takes between 5 and 10 seconds.**

get_dominant_topic_of_each_doc_as_df()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The method **get_dominant_topic_of_each_doc_as_df()** of the models.topics.TopicsModel class **is one of the most important method**,
because it generates a pandas DataFrame with the topic of each document of the sepcified dataset. That dataframe is used in many of
the TopicsModel methods. The problem with this method is that **it needs to predict the dominant topic of each document in the dataset**,
using the predict_topic_prob_on_text() method. As said above, the predict_topic_prob_on_text() takes 5-10 seconds with LdaMallet,
so **get_dominant_topic_of_each_doc_as_df() can be extremely slow**. For example, with the **20 newsgroups dataset**, and **10 seconds per prediction**,
this method takes **48 hours** to generate the DataFrame. But, **once generated, the DataFrame can be stored on disk** using the save_obj_to_disk()
method, and can be loaded in the __init__ method of the LdaMalletModel.

Lda Mallet Wrapper communication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The gensim **lda mallet wrapper communicates with the mallet library (written in Java) through temporary files**, so the
get_dominant_topic_of_each_doc_as_df() method can't be executed in parallel, because predictions are made using the same .txt file,
and that shared resource doens't leave to parallelize the predictions.

T-SNE
^^^^^
The **T-SNE** reduction and it's visualization uses the get_doc_topic_prob_matrix() method of the models.topics.TopicsModel class,
which is as slow as the get_dominant_topic_of_each_doc_as_df() method. Then, **LdaMalletModel is very slow to generate the
TSNE clustering chart**.

get_k_best_sentences_of_text()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The **get_k_best_sentences_of_text()** method of the models.summarization.TextRank class **sometimes can't converge** with
specific texts, throwing an PowerIterationFailedConvergence exception.

TwentyNewsGroupsDataset specific preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The **TwentyNewsGroupsDataset** class applies **specific preprocessing** to it's files, before the general preprocessing is applied.
This is because the documents of this dataset contain header and footer that tells too much information or unuseful information,
and also contain replies between users. All this info can be removed (and should be removed), setting the parameters of the
__init__ metthod to True (is the default value).





Extend the library
------------------

This section explains how to extend the library functionality.

Adding new datasets
^^^^^^^^^^^^^^^^^^^
The steps to add a new dataset are the following:

1. Create a new folder with the dataset files inside the **datasets folder**.
2. Create a python module with the following content:

   a. A <dataset-name>Dataset class
   b. A <dataset-name>Document class

Structured Dataset
""""""""""""""""""

If the dataset is divided in folders (like the 20 newsgroups dataset):

* The <dataset-name>Dataset class must inherit from the StructuredDataset class
* The <dataset-name>Document class must inherit from the StructuredDocument class

.. note:: The twenty_news_groups.py module can be used as an example.

Unstructured Dataset
""""""""""""""""""""

If the dataset is NOT divided in folders (instead, the dataset only consists on a folder with all the documents inside):

* A **unstructured_dataset.py** module must be created, with the UnstructuredDataset class, which inherits from the Dataset class.
  **The UnstructuredDataset must represent a dataset with a list of files.**
* The <dataset-name>Dataset class must inherit from the UnstructuredDataset class
* The <dataset-name>Document class must inherit from the Document class

Preprocessing unstructured datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If an unstructured dataset is created, preprocessing.dataset.py module must be converted into a package,
with the following modules:

* **preprocessing.dataset.structured.py**: with the same content of the current preprocessing.dataset.py module
* **preprocessing.dataset.unstructured.py**: with the same content of the current preprocessing.dataset.py module, but
  adapting all the functions that receive a StructuredDataset to use an UnstructuredDataset.

.. note:: This is because current preprocessing only can be used with StructuredDataset.
