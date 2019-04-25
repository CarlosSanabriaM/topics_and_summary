.. _introduction:

Introduction
=========================================================

The **objective** of this project was to create a tool that, given a set of documents,
performs the following actions:

* **Identify the topics present in the collection of documents.**
    A topic is a set of words that seem to be related, and seem to talk about an specific theme.
    For example, if the collection of documents comes from a news page, the topics
    obtained from that collection probably will be about religion, sports,
    technology, ... But the topic itself doesn't tells us the theme. It is described
    by the words contained in that topic. For example, a topic related with christianism
    may have the following words: god, jesus, christian, bible, ... With that words,
    we can identify that that topic talks about religion/christianism.

    .. image:: images/intro/wordcloud0.png
    <TODO footer>

* **Identify the relation of each document in the collection with each topic.**
    Each document will have a probability of being related with each topic.
    New texts can also be used to identify it's relation with the topics.

    .. image:: images/intro/predict-topics.png
    <TODO footer>

* **Classify each document in the collection inside a topic.**
    After identifying the relation of each document in the collection with
    each topic, the  obtained probability can be used to classify each document
    inside the topic with highest probability. New texts can also be used to
    classify them in a topic.

* **Create an extractive summary of a given document.**
    An extractive summary consists in a set of sentences directly extracted
    from the original text of the document.
    That sentences aim to summarize the whole content of the document.

With the functionality mentioned above, the **main utility** of this tool is to,
given a new text document, perform the following pipeline:

1. **Classify** the document into one of the obtained **topics**
2. Obtain an **extractive summary** of the document


Also, the following functionality was obtained extending this tool:

* **Obtain the most representative documents of each topic.**
    The most representative document of a topic will be the documents
    with the highest probability of being related with that topic.

* **Obtain the documents of the collection most related with a new text document.**
    The relation between the documents in the collection and the new texts can
    be obtained as follows:

    1. Obtain the topics more related with the given text.
    2. Obtain the documents more related with the topics obtained in step 1.
    3. Multiply the probability of the text being related with each topic
       by the probability of the most representative documents of that topic.
    4. Order the documents by the probability obtained in step 3.
