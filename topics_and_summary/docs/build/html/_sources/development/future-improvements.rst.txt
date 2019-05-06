Future Improvements
===================

Source code
-----------

* **Implement a solution for the TextRank main problem**: With some texts, this algorithm **doesn't converge**,
  throwing a PowerIterationFailedConvergence. Maybe in this case, the first k sentences of the text can be returned.
* **Evaluate the results of the TextRank summaries**: This can be done selecting some texts of the 20 newsgroups dataset,
  and obtaining theirs summaries with TextRank. Once we have the summaries, we can use a LdaMalletModel previously trained
  on that dataset to predict the topics probabilities of the summaries. After that, we can compare the topics probabilities
  of the original selected documents and the topics probabilities of their summaries. The comparisson of those probabilities
  can be used to evaluate the TextRank algorithm. We expect that the summary of a text relates to the same topics as the
  original text.
* **Implement more summarization approaches**. Some links to other alternatives can be found in the
  :ref:`other_summarization_alternatives` section of the Development: API page.
* **Revise the preprocessing order** of the preprocess_text() and preprocess_dataset() functions.
* **Add more datasets** to check the LdaMalletModel results on them, as explained in the :ref:`unstructured-dataset` section.
* **Implement the unstructured_dataset.py** module, as explained

.. note:: Evaluating text summaries is a very hard task, and most of the times it needs human expert intervention.
   There are some metrics, like ROUGE, but they also need an expert saying which sentences summarize better the text.

Documentation
-------------

* Change the **API packages and modules toctree** to don't show topics_and_summary as a first item
* Create equivalent .bat files for the existing .sh files used for generating the api documentation
* Create equivalent .bat files for the existing .sh files used for generating the requirements files