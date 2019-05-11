Quickstart
==========

The module **examples.demo.py** contains a **demo of the main functionality** of this library.

How to execute the demo
-----------------------

The demo can be executed in 2 ways:

a. If the library is installed, can be executed as follows:

   ::

      python
      >>> from topics_and_summary.examples import demo
      >>> demo.execute()

   .. warning:: Executing the demo this way sometimes leads to problems due to Python pickle objects stored in
      the saved-elements folder (that are used in the demo).

      This can happen if the library was installed like this: *pip install <project-root-path>*.

      A possible solution is to install the library this way: *pip install -e <project-root-path>*. This fixes the problem
      because it doesn't copy any files. Rather, it points to the source code folder. A broader explanation is present in the
      :ref:`development-installation-install-the-library` section of the Development Installation page.

      Another solution is to execute the demo without installing the library, as explained below.

b. If the library isn't installed, but the source code is available, it can be executed as follows:

   ::

      python
      >>> import sys
      >>> sys.path.extend(['<project-root-path>'])
      >>> exec(open("<project-root-path>/topics_and_summary/examples/demo.py").read())

Demo content
------------

This section explains the demo, that is divided in sections.

Section 1
^^^^^^^^^

**Loads the 20 newsgroups dataset and applies preprocessing**, or **loads a preprocessed dataset stored on disk**.

The TwentyNewsGroupsDataset.load() method loads a TwentyNewsGroupsDataset object previously stored on disk
using the save() method. The preprocessing options specified when the dataset was preprocessed are also loaded inside
the dataset object. This options will be used by the TopicsModel when receiving a text, because the text will be
preprocessed with the same options.

The ngrams='tri' option means that trigrams will be generated on words that appear together many times. For example,
'disk', 'operating' and 'system' appears together many times, so a trigram 'disk_operating_system' will be generated.

.. emphasize-lines counts from 1, not from the number specified in :lineno-start:
:emphasize-lines-in-source-code: 25,32,42 <-- -19+1 = 7,14,24
.. literalinclude:: ../../../examples/demo.py
    :linenos:
    :lineno-start: 19
    :language: python
    :lines: 19-44
    :emphasize-lines: 7,14,24



Section 2
^^^^^^^^^

**Generates a LdaGensimModel** or **loads a LdaMalletModel stored on disk**.

Both classes inherit from the TopicsModel class. The __init__ method of LdaGensimModel, LsaGensimModel and LdaMalletModel
creates the model. Creating the model in LdaMalletModel takes much more time than the others (up to 10 min).

LdaGensimModel, LsaGensimModel and LdaMalletModel have the load() method, that loads a previously created model,
stored on disk using the save() method. This method also loads the dataset used to generate the model,
the preprocessing options specified when the dataset was preprocessed, and the docs_topics_df DataFrame
(which contains the dominant topic of each document in the dataset).

The docs_topics_df pandas DataFrame was created with the method get_dominant_topic_of_each_doc_as_df().
This is a key method, because it's used by the other methods of the TopicsModel class. It generates a pandas
DataFrame with the most representative topic of each document in the dataset. To achieve this, it has to predict
the topics probabilities of each document. Each prediction can be done in less than a second with LdaGensimModel and
LsaGensimModel, but LdaMalletModel can take more than 5 seconds to make a prediction, so if the dataset has many
documents (as it should have), then get_dominant_topic_of_each_doc_as_df() is extremely slow with LdaMalletModel.

So, for this reason, the docs_topics_df is created once, and then it's stored on disk when the save() method of the
TopicsModel is called. After that, there is no need to recalculate the predictions.

.. emphasize-lines counts from 1, not from the number specified in :lineno-start:
:emphasize-lines-in-source-code: 53,54,58,61 <-- -48+1 = 6-10,12
.. literalinclude:: ../../../examples/demo.py
    :linenos:
    :lineno-start: 48
    :language: python
    :lines: 48-61
    :emphasize-lines: 6,7,11,14



Section 3
^^^^^^^^^

**Shows the topics** obtained with the model.

The topics can be showed in 2 ways:

* **Text format**, where each topic shows it's most important keywords, and the importance of each one inside the topic.
  This is done with the print_topics() method, that has a parameter pretty_format. If is True, topics are printed in a
  more structured way. If is false, each topic is printed in one line, as gensim does.
* **Wordclouds**, which saves some .png files with the most important keywords of each topic in the project root folder.
  This is done with the plot_word_clouds_of_topics() function of the visualizations module, that receives a List[Topic],
  that can be obtained with the get_topics() method of the TopicsModel class.

.. emphasize-lines counts from 1, not from the number specified in :lineno-start:
:emphasize-lines-in-source-code: 79,85-87 <-- -65+1 = 15,21-23
.. literalinclude:: ../../../examples/demo.py
    :linenos:
    :lineno-start: 65
    :language: python
    :lines: 65-87
    :emphasize-lines: 15,21-23



Section 4
^^^^^^^^^

**Shows the k most representative documents of the topic 16.**

This is done with the get_k_most_repr_docs_of_topic_as_df() method of the TopicsModel class.
This function uses the docs_topics_df. If the get_dominant_topic_of_each_doc_as_df() was called before in that model,
it is stored internally in the docs_topics_df attribute of the TopicsModel instance.

get_k_most_repr_docs_of_topic_as_df() returns a pandas DataFrame (ordered by document-topic probability),
with the k most representative documents of the specified topic.

.. emphasize-lines counts from 1, not from the number specified in :lineno-start:
:emphasize-lines-in-source-code: 100,105,110 <-- -91+1  = 10,15,20
.. literalinclude:: ../../../examples/demo.py
    :linenos:
    :lineno-start: 91
    :language: python
    :lines: 91-110
    :emphasize-lines: 10,15,20



Section 5
^^^^^^^^^

**Predicts the topics probabilities of a text.**

This is done with the predict_topic_prob_on_text() method of the TopicsModel class.
This method doesn't need the docs_topics_df. In fact, predict_topic_prob_on_text() is called by
the get_dominant_topic_of_each_doc_as_df() method, which is the one that generates the docs_topics_df.

This method is the one who communicates directly with the gensim models [#f1]_, calling the gensim model with the
indexing operation (self.model[text_as_bow]). TopicsModel is a wrapper of the gensim models functionality.

This method returns a pandas DataFrame, but it also can print a table with the results.

This method internally preprocess the given text, calling the preprocess_text() function of the
preprocessing.text module. The options passed to that function are the options stored in the preprocessing_options
attribute of the dataset, and are the same options as the ones selected when the dataset was preprocessed. If the
ngrams option was 'bi' or 'tri', the ngrams_model_func is also passed as one of the options (this function generates
the same bigrams/trigrams as the ones generated on the dataset documents).

The purpose of this is to apply the same preprocessing to new texts as the one applied to the dataset documents,
because if they are not preprocessed in the same way, the results won't be as expected.

.. emphasize-lines counts from 1, not from the number specified in :lineno-start:
:emphasize-lines-in-source-code: 154 <-- -114+1 = 41
.. literalinclude:: ../../../examples/demo.py
    :linenos:
    :lineno-start: 114
    :language: python
    :lines: 114-154
    :emphasize-lines: 41



Section 6
^^^^^^^^^

**Shows the k most related documents to a text**

This is done with the get_related_docs_as_df() method of the TopicsModel class.
This method calls the predict_topic_prob_on_text() and the  get_k_most_repr_docs_per_topic_as_df() methods,
so it needs the docs_topics_df.

.. emphasize-lines counts from 1, not from the number specified in :lineno-start:
:emphasize-lines-in-source-code: 170,175,190 <-- -158+1 = 13,18,23
.. literalinclude:: ../../../examples/demo.py
    :linenos:
    :lineno-start: 158
    :language: python
    :lines: 158-180
    :emphasize-lines: 13,18,23



Section 7
^^^^^^^^^

**Summarizes a text**

This is done with the TextRank class (that uses the TextRank algorithm). This class has only one method:
get_k_best_sentences_of_text(), that returns the k best sentences of the given text
(the ones that summarizes it the most). It performs, therefore, an extractive summary.

Internally, the get_k_best_sentences_of_text() method uses word embeddings (either GloVe or Word2Vec).

.. emphasize-lines counts from 1, not from the number specified in :lineno-start:
:emphasize-lines-in-source-code: 197,199 <-- -184+1 = 14,16
.. literalinclude:: ../../../examples/demo.py
    :linenos:
    :lineno-start: 184
    :language: python
    :lines: 184-204
    :emphasize-lines: 14,16




Execute the demo using a docker container
------------------------------------------

A docker image can be created using the Dockerfile. **Docker must be installed.** See the `docker web page <https://www.docker.com>`__.

This allows to install all the dependencies in a separeted "enviroment" (a docker container).

.. warning:: The Dockerfile expects that the **20 newsgroups dataset, the glove embeddings, and the mallet source
   have been download by the user**.

   * The topics_and_summary/datasets folder must contain the 20_newsgroups.
   * The topics_and_summary/embeddings folder must contain the glove folder.
   * The mallet-2.0.8 folder must contain the mallet source code.

   The :ref:`download-other-elements` section of the Usage Installation page explains how to download this files.

   The **word2vec embeddings are not needed**, because they are not used in the demo.

The steps are the following:

1. Create a docker image using the Dockerfile
2. Create a docker container using the recently created image
3. Run the docker container

Creating the docker image
^^^^^^^^^^^^^^^^^^^^^^^^^

Execute the following commands to create the docker image:

   ::

      cd <project-root-path>
      docker build . -t topics_and_summary:latest

.. note:: This is the step that takes most time to execute. 10-20 minutes, depending on your system.

Creating and running a docker container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Execute the following commands to create and run a docker container using the previously created image:

::

   docker run --name topics_and_summary -v $PWD/demo-images:/topics_and_summary/demo-images -i -t topics_and_summary:latest

After executing this command, the demo will start. When the demo finishes, the container will stop.
To execute the demo more than one time, see the section below.

.. note:: This step is much faster than the previous one.

Running an existing docker container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the docker container was previously created, to execute the demo again you don't need to create another container.
You can start the existing container, using the following command:

::

   docker start -i topics_and_summary

Removing the docker container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To remove the docker container, execute the following command:

::

   docker container rm topics_and_summary

Removing the docker image
^^^^^^^^^^^^^^^^^^^^^^^^^

To remove the docker image, execute the following command:

::

   docker image rm topics_and_summary:latest





.. rubric:: Footnotes

.. [#f1] Other methods also communicate directly with the gensim models, but this specific communication is the most
   important one, because requests the gensim model to make predictions of the topics probabilities of a given text.
