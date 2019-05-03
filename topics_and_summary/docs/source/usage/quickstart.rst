Quickstart
==========

Demo
----
The module **examples.demo.py** contains a **demo of the main functionality** of this library.

How to execute it
^^^^^^^^^^^^^^^^^

The demo can be executed in 2 ways:

a. If the library is installed, can be executed as follows:

   ::

      python
      >>> from topics_and_summary.examples import demo
      >>> demo.execute()

a. If the library source code is available, it also can be executed as follows:

   ::

      python <project-root-path>/topics_and_summary/examples/demo.py

Demo content
^^^^^^^^^^^^

This section explains the demo, which is divided in sections.

Section 1
"""""""""

**Loads the 20 newsgroups dataset and applies preprocessing**, or **loads a preprocessed dataset stored on disk**.

The trigrams_func is the function obtained after preprocessing the dataset with the ngrams='tri' option.
It needs to be saved, because it will be used later to preprocess the text passed to the model to predict the
topics probability on that text, or to return the related documents.

.. emphasize-lines counts from 1, not from the number specified in :lineno-start:
   :emphasize-lines-in-source-code: 21,22,27,35 <-- -16+1  = 6,7,12,18
.. literalinclude:: ../../../examples/demo.py
    :linenos:
    :lineno-start: 16
    :language: python
    :lines: 16-37
    :emphasize-lines: 6,7,12,18



Section 2
"""""""""

**Generates a LdaGensimModel** or **loads a LdaMalletModel stored on disk**.

Both classes inherit from the TopicsModel class. The __init__ method of LdaGensimModel, LsaGensimModel and LdaMalletModel
creates the model. Creating the model in LdaMalletModel takes much more time than the others (up to 10 min).

LdaGensimModel, LsaGensimModel and LdaMalletModel have the load() method, that loads a previously created model,
stored on disk using the save() method.

docs_topics_df is a pandas DataFrame that was created with the method get_dominant_topic_of_each_doc_as_df().
This is a key method, because it's used by the others. It generates a pandas DataFrame with the most representative topic
of each document in the dataset. To achieve this, it has to predict the topics probabilities of each document.
Each prediction can be done in less than a second with LdaGensimModel and LsaGensimModel, but LdaMalletModel can take more than
5 seconds to make a prediction, so if the dataset has many documents (as it should have), then get_dominant_topic_of_each_doc_as_df() is
extremely slow with LdaMalletModel.

So, for this reason, the docs_topics_df is created once, and then it's stored on disk. After that, there is no need
to recalculate the predictions. The only thing that needs to be done is to load in memory the docs_topics_df stored on disk,
and pass it to the load() or __init__ method of the models.

.. emphasize-lines counts from 1, not from the number specified in :lineno-start:
   :emphasize-lines-in-source-code: 46-50, 52 <-- -41+1  = 6-10,12
.. literalinclude:: ../../../examples/demo.py
    :linenos:
    :lineno-start: 41
    :language: python
    :lines: 41-52
    :emphasize-lines: 6-10,12



Section 3
"""""""""

**Shows the topics** obtained with the model.

The topics can be showed in 2 ways:

* **Text format**, where each topic shows it's most important keywords, and the importance of each one inside the topic.
  This is done with the print_topics(), which has a parameter pretty_format. If is True, topics are printed in a
  more structured way. If is false, each topic is printed in one line, as gensim does.
* **Wordclouds**, which plots the most important keywords of each topic in a matplotlib.pyplot plot. This is done with
  the plot_word_clouds_of_topics() function of the visualizations module, which receives a List[Topic], that can be
  obtained with the get_topics() method of the TopicsModel class.

.. emphasize-lines counts from 1, not from the number specified in :lineno-start:
   :emphasize-lines-in-source-code: 70,73 <-- -56+1  = 15,18
.. literalinclude:: ../../../examples/demo.py
    :linenos:
    :lineno-start: 56
    :language: python
    :lines: 56-73
    :emphasize-lines: 15,18



Section 4
"""""""""

**Shows the k most representative documents of the topic 16.**

This is done with the get_k_most_repr_docs_of_topic_as_df() method of the TopicsModel class.
This function uses the docs_topics_df. If the get_dominant_topic_of_each_doc_as_df() was called before, it is stored
internally in the TopicsModel instance. The docs_topics_df could also have been loaded from disk.

get_k_most_repr_docs_of_topic_as_df() returns a pandas DataFrame (ordered by document-topic probability),
with the k most representative documents of the specified topic.

.. emphasize-lines counts from 1, not from the number specified in :lineno-start:
   :emphasize-lines-in-source-code: 85,88,90 <-- -77+1  = 9,12,14
.. literalinclude:: ../../../examples/demo.py
    :linenos:
    :lineno-start: 77
    :language: python
    :lines: 77-90
    :emphasize-lines: 9,12,14



Section 5
"""""""""

**Predicts the topics probabilities of a text.**

This is done with the predict_topic_prob_on_text() method of the TopicsModel class.
This method doesn't need the docs_topics_df. In fact, predict_topic_prob_on_text() is called by
the get_dominant_topic_of_each_doc_as_df() method, which generates the docs_topics_df.

This method is the one who communicates directly with the gensim models [#f1]_, calling the gensim model with the
indexing operation (self.model[text_as_bow]). TopicsModel is a wrapper of the gensim models functionality.

This method returns a pandas DataFrame, but it also can print a table with the results.

In this example, be can see that the *ngrams='tri'* and the *ngrams_model_func=trigrams_func* parameters are passed.
This is because this method internally preprocess the dataset, calling the preprocess_dataset() function of the
preprocessing.dataset module. The option *ngrams='tri'* tells the method to generate trigrams on the given text,
using for that the *ngrams_model_func* passed as a parameter. This options are passed because the topics model was
generated with a dataset preprocessed generating trigrams, using the preprocess_dataset() function of the
preprocessing.dataset module. The preprocess_dataset() function returned the preprocessed dataset and a trigrams_func,
which is the one passed here to the ngrams_model_func parameter.

.. emphasize-lines counts from 1, not from the number specified in :lineno-start:
   :emphasize-lines-in-source-code: 119 <-- -94+1  = 26
.. literalinclude:: ../../../examples/demo.py
    :linenos:
    :lineno-start: 94
    :language: python
    :lines: 94-119
    :emphasize-lines: 26



Section 6
"""""""""

**Shows the k most related documents to a text**

This is done with the get_related_docs_as_df() method of the TopicsModel class.
This method calls the predict_topic_prob_on_text() and the  get_k_most_repr_docs_per_topic_as_df() methods,
so it needs the docs_topics_df.

In this example, the *ngrams='tri'* and the *ngrams_model_func=trigrams_func* parameters are passed again,
for the same reason as the one explained above.

.. emphasize-lines counts from 1, not from the number specified in :lineno-start:
   :emphasize-lines-in-source-code: 134-136,139,141 <-- -123+1 = 12-14,17,19
.. literalinclude:: ../../../examples/demo.py
    :linenos:
    :lineno-start: 123
    :language: python
    :lines: 123-141
    :emphasize-lines: 12-14,17,19



Section 7
"""""""""

**Summarizes a text**

This is done with the TextRank class (that uses the TextRank algorithm). This class has only one method:
get_k_best_sentences_of_text(), which returns the k best sentences of the given text
(the ones that summarizes it the most). It performs, therefore, an extractive summary.

Internally, the get_k_best_sentences_of_text() method uses word embeddings (either GloVe or Word2Vec).

.. emphasize-lines counts from 1, not from the number specified in :lineno-start:
   :emphasize-lines-in-source-code: 157,158 <-- -145+1 = 13,14
.. literalinclude:: ../../../examples/demo.py
    :linenos:
    :lineno-start: 145
    :language: python
    :lines: 145-162
    :emphasize-lines: 13,14



.. rubric:: Footnotes

.. [#f1] Other methods also communicate directly with the gensim models, but this specific communication is the most
   important one, because requests the gensim model to make predictions of the topics probabilities of a given text.