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

  .. note:: Evaluating text summaries is a very hard task, and most of the times it needs human expert intervention.
     There are some metrics, like ROUGE, but they also need an expert saying which sentences summarize better the text.

* **Implement more summarization approaches**. Some links to other alternatives can be found in the
  :ref:`other_summarization_alternatives` section of the Development: API page.
* **Revise the preprocessing order** of the preprocess_text() and preprocess_dataset() functions.
* **Add more datasets** to check the LdaMalletModel results on them, as explained in the :ref:`unstructured-dataset` section.
* **Implement the unstructured_dataset.py** module, as explained in the :ref:`unstructured-dataset` section of the *Development: API* page.
* **Extract all the paths and constant values in the source code to a configuration file**, similar to what the demo does
  with the demo-conf.ini file.
* **Modify the .dockerignore file** to ignore everything at the beginning and then include all the needed files.
* **Find a way to use the Java mallet library without all the source code.** A Java JAR file should be created.
* **Change the global .gitignore and use .gitkeep files** instead of including a .gitignore file in the empty folders
  that need to be kept in git.
* Create equivalent .bat files for the existing .sh files.



Documentation
-------------

* **Fix the following warning**: *WARNING: Explicit markup ends without a blank line; unexpected unindent.*
  This warning is caused by the comments. The second line of the comments need to be indented with at least 2 spaces,
  but the PyCharm IDE removes those spaces after reformating the code.
* Change the **API packages and modules toctree** to don't show *topics_and_summary package* as a first item.
* Generate the documentation of each Python module or class in the same way as pandas library documentation does.
  As an example, check the `pandas Series class documentation <https://pandas.pydata.org/pandas-docs/stable/reference/series.html>`__.
  This is done with the \.. autosummary:: directive. The soure code of the pandas Series class documentation
  is available on `this GitHub page <https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/source/reference/series.rst>`__.
* Create equivalent .bat files for the existing .sh files used for generating the api documentation.
* Create equivalent .bat files for the existing .sh files used for generating the requirements files.
