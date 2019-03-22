import abc
import os

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from texttable import Texttable
from tqdm import tqdm

from preprocessing.text import preprocess_text
from utils import get_abspath, RANDOM_STATE, now_as_str, join_paths


def prepare_corpus(documents):
    """
    Given a list of documents, returns a term dictionary (dictionary) and a document-term matrix (corpus).
    :param documents: List of documents. Each document is a list of words, where each word is a string.
    :return: 2 things: Term's dictionary (it's attribute token2id is a dict mapping words to numeric identifiers)
    and Document-term matrix (a list, where each element is a list of tuples. Each tuple contains the index of a word
    and the number of times that word appears in that document).
    """
    # Creating the term dictionary of our corpus, where every unique term is assigned an index.
    dictionary = gensim.corpora.Dictionary(documents)
    # Converting list of documents into Document Term Matrix using dictionary prepared above.
    corpus = get_corpus(dictionary, documents)

    return dictionary, corpus


def get_corpus(dictionary, documents):
    """
    Returns a corpus/document-term matrix, that consists on a list, where each element is a list of tuples.
    Each list of tuples represents a document, and each tuple contains the index of a word in that document
    and the number of times that word appears in that document.
    :param dictionary: gensim.corpora.Dictionary object.
    :param documents: List of lists of strings. Each one of the nested lists represents a document, and the
    strings the words in that document.
    :return: Returns a corpus/document-term matrix, that consists on a list, where each element is a list of tuples.
    """
    return [dictionary.doc2bow(doc) for doc in documents]


class TopicsModel(metaclass=abc.ABCMeta):
    """Base class for a single topics model."""

    __SAVE_PATH = '../saved-models/topics/'  # Path where the models will be saved

    def __init__(self, documents, dictionary=None, corpus=None, num_topics=20, model=None, **kwargs):
        """
        :param documents: List of lists of strings. Each one of the nested lists represents a document,
        and the strings the words in that document.
        :param dictionary: gensim.corpora.Dictionary object. If is None, it is created using the documents.
        :param corpus: Document-term matrix. If is None, it is created using the documents.
        :param num_topics: Number of topics.
        :param model: Pre-created model. If is None, a model is created.
        :param kwargs: Additional arguments.
        """
        self.documents = documents
        self.num_topics = num_topics
        self.coherence_value = None
        self.docs_topics_df = None
        self.dir_path = None  # Path of the directory where the model is saved to

        if dictionary is None or corpus is None:
            self.dictionary, self.corpus = prepare_corpus(documents)
        else:
            self.dictionary, self.corpus = dictionary, corpus

        if model is None:
            self.model = self._create_model(**kwargs)
        else:
            self.model = model

    @abc.abstractmethod
    def _create_model(self, num_topics, **kwargs):
        """
        Factory Method design pattern. The subclasses override this method,
        creating and returning the specific model that the subclasses represent.
        :param num_topics: Number of topics of the model.
        """

    def compute_coherence_value(self, coherence='c_v'):
        """
        Calculates, stores and returns the coherence value of the topics model.
        :param coherence: String that represents the type of coherence to calculate.
        Valid values are: ‘c_v’, ‘c_uci’ and ‘c_npmi’.
        :return: The coherence value of the topics model.
        """
        # Create the coherence model and evaluate the model
        coherence_model = gensim.models.CoherenceModel(model=self.model, texts=self.documents,
                                                       dictionary=self.dictionary, coherence=coherence)
        self.coherence_value = coherence_model.get_coherence()

        return self.coherence_value

    def save(self, base_name, path=__SAVE_PATH):
        """
        Saves the model to disk.
        :param base_name: Base name of the model. After it, the current time, the number of topics, and
        the coherence value are added.
        :param path: Path were the models will be stored.
        and the coherence value of the models is added.
        """
        if self.coherence_value is None:
            self.compute_coherence_value()

        now = now_as_str()
        model_name = "{0}_{1}topics_coherence{2}_{3}".format(base_name, str(self.model.num_topics),
                                                             str(self.coherence_value), now)

        self.dir_path = get_abspath(__file__, join_paths(path, model_name))
        os.mkdir(self.dir_path)
        model_path = join_paths(self.dir_path, model_name)
        self.model.save(model_path)

        # Save the coherence value in a .txt file
        coherence_path = join_paths(self.dir_path, "coherence_value.txt")
        with open(coherence_path, 'w') as f:
            f.write(str(self.coherence_value))

    @classmethod
    def load(cls, model_name, documents, model_dir_path=__SAVE_PATH):
        """
        Loads the model with the given name from the specified path, and
        returns a TopicsModel instance.
        :param model_dir_path: Path to the directory where the model is in.
        :param model_name: Model name.
        :param documents:
        :return:
        """
        model = cls._load_gensim_model(join_paths(model_dir_path, model_name) + "/" + model_name)
        return cls(documents, num_topics=model.num_topics, model=model, model_name=model_name)

    @classmethod
    @abc.abstractmethod
    def _load_gensim_model(cls, path):
        """
        Factory Method design pattern. The subclasses override this method,
        loading the gensim model in the specified path and returning it.
        :param path: Path of the saved gensim model.
        :return: The gensim model.
        """

    def print_topics(self, num_words_each_topic=10, num_topics=None):
        """
        Prints the topics of the topics model.
        :param num_words_each_topic: Number of words of each topic to print.
        :param num_topics: Number of topics to show. If not specified, default is the num topics in the lsa model.
        """
        if num_topics is None:
            num_topics = self.model.num_topics

        # Sequence with (topic_id, [(word, value), ... ]).
        topics_sequence = self.model.print_topics(num_topics=num_topics, num_words=num_words_each_topic)
        for topic in topics_sequence:
            print('Topic ' + str(topic[0]) + ': ' + topic[1])

    def predict_topic_prob_on_text(self, text, num_best_topics=None, preprocess=True, print_table=True):
        """
        Predicts the probability of each topic to be related to the given text.
        The probabilities sum 1. When the probability of a topic is very high, the other
        topics may not appear in the results.
        :param text: Text.
        :param num_best_topics: Number of topics to return. If is None, returns all the topics that the model returns.
        :param preprocess: If true, applies preprocessing to the given text using preprocessing.text.preprocess_text().
        :param print_table: If True, this method also prints a table with the topics indices,
        their probabilities, and their keywords.
        :return: Topic probability vector.
        """
        if preprocess:
            text = preprocess_text(text)

        text_as_bow = self.dictionary.doc2bow(text.split())
        topic_prob_vector = self.model[text_as_bow]
        topic_prob_vector = sorted(topic_prob_vector, key=lambda x: x[1], reverse=True)

        if num_best_topics is None:
            num_best_topics = len(topic_prob_vector)

        if print_table:
            table = Texttable()
            table.set_cols_width([10, 10, 80])
            table.set_cols_align(['c', 'c', 'l'])

            # Specify header
            table.set_header_align(['c', 'c', 'c'])
            table.header(['Topic index', 'Topic prob', 'Topic keywords'])

            for topic_prob_pair in topic_prob_vector[:num_best_topics]:
                topic_index = topic_prob_pair[0]
                topic_prob = topic_prob_pair[1]
                table.add_row([topic_index, topic_prob, self.model.print_topic(topic_index)])

            print(table.draw())

        return topic_prob_vector[:num_best_topics]

    def get_related_documents_as_df(self, text, k_docs_per_topic=1, preprocess=True):
        """
        Given a text, this method returns a df with the index and the content of the most similar documents
        in the corpus. The similar/related documents are obtained as follows:
        1. Obtain the topics more related with the given text.
        2. Obtain the documents more related with the topics obtained in step 1.
        The returned df contains the documents indices, it's content, and the probability of that document
        being related with the given text (with the topics and the documents we have).
        That probability is obtained as follows: Probability of the text being related with the topic * Probability
        that the document influence the topic.
        :param text: String.
        :param k_docs_per_topic: Number of documents per topic to be used to retrieve the related documents.
        :param preprocess: If True, apply preprocessing to the text.
        :return: The pandas DataFrame.
        """
        # 1. Obtain the list of topics more related with the text
        topic_prob_vector = self.predict_topic_prob_on_text(text, preprocess=preprocess, print_table=False)
        topics = list(map(lambda x: x[0], topic_prob_vector))
        # TODO: If the prob between the first topic and the second is very high, maybe is better to use only the first
        #  topic and forget about the rest. The number of topics to return will be k_docs_per_topic * len(topics),
        #  although the docs to return will be only of the first topic.

        # 2. Obtain a df with the documents more related with the topics in the previous step
        k_most_repr_doc_per_topic_df = self.get_k_most_representative_docs_per_topic_as_df(k=k_docs_per_topic)
        related_docs_df = k_most_repr_doc_per_topic_df.loc[k_most_repr_doc_per_topic_df['Topic index'].isin(topics)]

        # 3. Transform the df to have the following columns: Doc index, Doc prob, Doc text
        # Doc prob = prob of the text being related with the topic * prob that the doc influence the topic

        def get_prob_of_topic(topic_index):
            return next(filter(lambda x: x[0] == topic_index, topic_prob_vector))[1]

        # Iterate for each row, get the prob text-topic from the topic_prob_vector using the 'Topic index' of that row,
        # get the prob doc-topic from that row, and multiply them.
        doc_prob_column = related_docs_df.apply(lambda row:
                                                get_prob_of_topic(row['Topic index']) * row['Topic prob'],
                                                axis='columns')

        # Remove other columns
        # TODO: Maybe it's interesting to keep this 3 columns to give the user info about why that doc is related
        #  to the given text. This columns can be moved to the end of the df.
        related_docs_df = related_docs_df.drop(columns=['Topic index', 'Topic prob', 'Topic keywords'])

        # Add the 'Doc prob' column
        related_docs_df.insert(2, 'Doc prob', doc_prob_column, allow_duplicates=True)

        # Change columns order
        related_docs_df = related_docs_df[['Doc index', 'Doc prob', 'Doc text']]

        # Order by 'Doc prob' column in descending order
        related_docs_df = related_docs_df.sort_values(['Doc prob'], ascending=[False])

        # Reset the indices
        related_docs_df.reset_index(drop=True, inplace=True)

        return related_docs_df

    def get_dominant_topic_of_each_doc_as_df(self):
        """
        Returns a pandas DataFrame with the following columns: Doc index, Dominant topic index, Topic prob,
        Topic keywords, Doc text. This method can take to much time to execute if the dataset is big.
        :return: pandas DataFrame.
        """

        # Iteratively appending rows to a DataFrame can be more computationally intensive than a single concatenate.
        # A better solution is to append those rows to a list and then concatenate the list with the original
        # DataFrame all at once.
        rows_list = []

        for doc_index, doc_as_bow in enumerate(tqdm(self.corpus)):
            dominant_topic = sorted(self.model[doc_as_bow], key=lambda x: x[1], reverse=True)[0]
            dominant_topic_index = dominant_topic[0]
            dominant_topic_prob = dominant_topic[1]
            dominant_topic_kws = self.model.show_topic(dominant_topic_index)

            rows_list.append(
                pd.DataFrame([
                    [doc_index, dominant_topic_index, dominant_topic_prob, dominant_topic_kws,
                     ' '.join(self.documents[doc_index])]
                ], columns=['Doc index', 'Dominant topic index', 'Topic prob', 'Topic keywords', 'Doc text'])
            )

        # Concat the dfs to a one df, store it, and return it
        self.docs_topics_df = pd.concat(rows_list)
        self.docs_topics_df.reset_index(drop=True, inplace=True)
        return self.docs_topics_df

    def get_k_most_representative_docs_per_topic_as_df(self, k=1):
        """
        Returns a DataFrame where the topics are grouped in ascending order by their indices, and inside each
        topic group there are k rows, where each row contains the topic and one of the most representative documents
        of that topic, in descending order.
        get_dominant_topic_of_document_each_doc_as_df(). If is None, that method is call, and that can take be slow.
        :param k: Number of the most representative documents per topic you want.
        :return: A pandas DataFrame with the following columns: Topic index, Doc index, Topic prob, Topic keywords and
        Doc text.
        """
        if self.docs_topics_df is None:
            self.get_dominant_topic_of_each_doc_as_df()

        k_most_repr_doc_per_topic_df = pd.DataFrame()

        # Group rows by the topic index
        doc_topics_grouped_by_topic_df = self.docs_topics_df.groupby('Dominant topic index')

        # For each topic group, sort the docs by the 'Topic prob' (in descending order) and select the first k ones
        for topic, group in doc_topics_grouped_by_topic_df:
            most_repr_docs = group.sort_values(['Topic prob'], ascending=[False]).head(k)
            k_most_repr_doc_per_topic_df = pd.concat([k_most_repr_doc_per_topic_df, most_repr_docs],
                                                     axis=0)

        # Reset indices
        k_most_repr_doc_per_topic_df.reset_index(drop=True, inplace=True)
        # Change columns names
        k_most_repr_doc_per_topic_df.columns = ['Doc index', 'Topic index', 'Topic prob', 'Topic keywords', 'Doc text']
        # Change columns order
        k_most_repr_doc_per_topic_df = \
            k_most_repr_doc_per_topic_df[['Topic index', 'Doc index', 'Topic prob', 'Topic keywords', 'Doc text']]
        # Order rows by topic index (in ascending order). Each topic group is ordered by the Topic prob, so no problem.
        k_most_repr_doc_per_topic_df = k_most_repr_doc_per_topic_df.sort_values(['Topic index'], ascending=[True])

        return k_most_repr_doc_per_topic_df

    def get_topic_distribution_as_df(self):
        """
        Returns a DataFrame where each row contains a topic, the number of documents of that topic (the topic
        is the dominant topic of those documents), and the percentage of documents of that topic.
        :return: A pandas DataFrame with the following columns: Dominant topic index, Num docs, Percentage docs and
        Topic keywords.
        """
        if self.docs_topics_df is None:
            self.get_dominant_topic_of_each_doc_as_df()

        # DataFrame of docs and topics grouped by topic
        docs_topics_grpd_by_topic_df = self.docs_topics_df.groupby('Dominant topic index')
        # Number of docs per topic
        topic_docs_count = docs_topics_grpd_by_topic_df[['Doc index']].count()
        # Percentage of docs per topic
        topic_docs_perc = topic_docs_count / topic_docs_count.sum()

        # Concat the previous columns into a single df
        df_dominant_topics = pd.concat([topic_docs_count, topic_docs_perc], axis=1)
        df_dominant_topics.reset_index(inplace=True)

        # Obtain Topic keywords and add a column with them to the df
        topic_kws = pd.Series([self.model.show_topic(i) for i in range(20)])
        df_dominant_topics = pd.concat([df_dominant_topics, topic_kws], axis=1)

        # Change Column names
        df_dominant_topics.columns = ['Dominant topic index', 'Num docs', 'Percentage docs', 'Topic keywords']

        return df_dominant_topics

    def get_doc_topic_prob_matrix(self):
        """
        :return: Returns a numpy matrix where the rows are documents and columns are topics.
        Each cell represents the probability of the document in that row being related with the topic in that column.
        """
        num_docs = len(self.documents)
        num_topics = self.num_topics
        doc_topic_prob_matrix = np.zeros((num_docs, num_topics))

        for i in tqdm(range(doc_topic_prob_matrix.shape[0])):
            weights = self.model[self.corpus[i]]
            for topic_index, topic_prob in weights:
                doc_topic_prob_matrix.itemset((i, topic_index), topic_prob)

        return doc_topic_prob_matrix

    def get_k_kws_per_topic_as_str(self, topic, k):
        """
        :param k: Num keywords per topic.
        :return: k keywords from the given topic as a str.
        """
        return ' '.join(list(map(lambda x: x[0], self.model.show_topic(topic)))[:k])


class LdaMalletModel(TopicsModel):
    """Class that encapsulates the functionality of gensim.models.wrappers.LdaMallet, making it easier to use."""

    __MALLET_SOURCE_CODE_PATH = '../../../mallet-2.0.8/bin/mallet'
    __MALLET_SAVED_MODELS_PATH = '../saved-models/topics/lda_mallet'

    def __init__(self, documents, dictionary=None, corpus=None, num_topics=20,
                 model=None, mallet_path=get_abspath(__file__, __MALLET_SOURCE_CODE_PATH),
                 model_name=None, model_path=__MALLET_SAVED_MODELS_PATH, **kwargs):
        """
        Encapsulates the functionality of gensim.models.wrappers.LdaMallet, making it easier to use.
        :param documents: List of lists of strings. Each one of the nested lists represents a document,
        and the strings the words in that document.
        :param dictionary: gensim.corpora.Dictionary object. If is None, it is created using the documents.
        :param corpus: Document-term matrix. If is None, it is created using the documents.
        :param num_topics: Number of topics.
        :param model: Pre-created model. If is None, a model is created.
        :param mallet_path: Path to the mallet source code.
        :param model_name: Name of the folder where all mallet files will be saved. That folder will be created
        inside model_path directory. This param is obligatory.
        :param model_path: Path of the directory where the mallet directory will be created.
        :param kwargs: Additional keyword arguments that want to be used in the gensim.models.wrappers.LdaMallet
        __init__ method.
        """
        if model_name is None:
            raise ValueError('model_name parameter is obligatory')

        self.model_name = model_name

        super().__init__(documents, dictionary, corpus, num_topics, model,
                         mallet_path=mallet_path, model_path=model_path, **kwargs)

    def _create_model(self, **kwargs):
        """
        Creates the LdaMallet model and returns it.
        :param kwargs: Keyword arguments that want to be used in the gensim.models.wrappers.LdaMallet.
        mallet_path argument is obligatory.
        :return: The gensim.models.wrappers.LdaMallet model created.
        """
        # Create the folder where the mallet files will be stored

        prefix = get_abspath(__file__, join_paths(kwargs['model_path'], self.model_name))
        del kwargs['model_path']  # Remove it to avoid passing it to the LdaMallet __init__ method above
        os.mkdir(prefix)

        self.dir_path = prefix

        # Add the model_name again to the prefix. Mallet expects the prefix to be the name of a file
        # inside the folder where the mallet files will be saved. For that reason, we add to the folder
        # path the model name, and we will use later the prefix to store the final model.
        prefix = join_paths(prefix, self.model_name)

        return gensim.models.wrappers.LdaMallet(corpus=self.corpus,
                                                id2word=self.dictionary,
                                                num_topics=self.num_topics,
                                                prefix=prefix,
                                                **kwargs)  # mallet_path is passed here

    @classmethod
    def _load_gensim_model(cls, path):
        """
        Loads the gensim.models.wrappers.LdaMallet in the specified path and returns it.
        :param path: Path of the saved gensim.models.wrappers.LdaMallet.
        :return: The gensim.models.wrappers.LdaMallet model.
        """
        return gensim.models.wrappers.LdaMallet.load(get_abspath(__file__, path))

    # noinspection PyMethodOverriding
    def save(self):
        """
        Saves the mallet model to the path specified in self.model.prefix and stores in a .txt the coherence value.
        """
        # Save the model and all it's files
        self.model.save(self.model.prefix)

        # Save the coherence value in a .txt file
        if self.coherence_value is None:
            self.compute_coherence_value()

        coherence_path = join_paths(self.dir_path, "coherence_value.txt")
        with open(coherence_path, 'w') as f:
            f.write(str(self.coherence_value))

    @classmethod
    def load(cls, model_name, documents, model_dir_path=__MALLET_SAVED_MODELS_PATH):
        return super(LdaMalletModel, cls).load(model_name, documents, model_dir_path)


class LdaGensimModel(TopicsModel):
    """Class that encapsulates the functionality of gensim.models.LdaModel, making it easier to use."""

    __LDA_SAVED_MODELS_PATH = '../saved-models/topics/lda/'

    def __init__(self, documents, dictionary=None, corpus=None, num_topics=20,
                 model=None, random_state=RANDOM_STATE, **kwargs):
        """
        Encapsulates the functionality of gensim.models.LdaModel, making it easier to use.
        :param documents: List of lists of strings. Each one of the nested lists represents a document,
        and the strings the words in that document.
        :param dictionary: gensim.corpora.Dictionary object. If is None, it is created using the documents.
        :param corpus: Document-term matrix. If is None, it is created using the documents.
        :param num_topics: Number of topics.
        :param model: Pre-created model. If is None, a model is created.
        :param random_state: Random state for reproducibility.
        :param kwargs: Additional keyword arguments that want to be used in the gensim.models.LdaModel __init__ method.
        """
        super().__init__(documents, dictionary, corpus, num_topics, model, random_state=random_state, **kwargs)

    def _create_model(self, **kwargs):
        """
        Creates the Lda model and returns it.
        :param kwargs: Keyword arguments that want to be used in the gensim.models.LdaModel.
        random_state argument is obligatory.
        :return: The gensim.models.LdaModel model created.
        """
        return gensim.models.LdaModel(corpus=self.corpus,
                                      id2word=self.dictionary,
                                      num_topics=self.num_topics,
                                      **kwargs)  # random_state is passed here

    def save(self, base_name, path=__LDA_SAVED_MODELS_PATH):
        super(LdaGensimModel, self).save(base_name, path)

    @classmethod
    def load(cls, model_name, documents, model_dir_path=__LDA_SAVED_MODELS_PATH):
        return super(LdaGensimModel, cls).load(model_name, documents, model_dir_path)

    @classmethod
    def _load_gensim_model(cls, path):
        """
        Loads the gensim.models.LdaModel in the specified path and returns it.
        :param path: Path of the saved gensim.models.LdaModel.
        :return: The gensim.models.LdaModel.
        """
        return gensim.models.LdaModel.load(get_abspath(__file__, path))


class LsaGensimModel(TopicsModel):
    """Class that encapsulates the functionality of gensim.models.LsiModel, making it easier to use."""

    __LSA_SAVED_MODELS_PATH = '../saved-models/topics/lsa/'

    def __init__(self, documents, dictionary=None, corpus=None, num_topics=20, model=None, **kwargs):
        """
        Encapsulates the functionality of gensim.models.LsiModel, making it easier to use.
        :param documents: List of lists of strings. Each one of the nested lists represents a document,
        and the strings the words in that document.
        :param dictionary: gensim.corpora.Dictionary object. If is None, it is created using the documents.
        :param corpus: Document-term matrix. If is None, it is created using the documents.
        :param num_topics: Number of topics.
        :param model: Pre-created model. If is None, a model is created.
        :param kwargs: Additional keyword arguments that want to be used in the gensim.models.LsiModel __init__ method.
        """
        super().__init__(documents, dictionary, corpus, num_topics, model, **kwargs)

    def _create_model(self, **kwargs):
        """
        Creates the Lda model and returns it.
        :param kwargs: Keyword arguments that want to be used in the gensim.models.LdaModel.
        :return: The gensim.models.LsiModel model created.
        """
        return gensim.models.LsiModel(corpus=self.corpus,
                                      id2word=self.dictionary,
                                      num_topics=self.num_topics,
                                      **kwargs)

    @classmethod
    def load(cls, model_name, documents, model_dir_path=__LSA_SAVED_MODELS_PATH):
        return super(LsaGensimModel, cls).load(model_name, documents, model_dir_path)

    def save(self, base_name, path=__LSA_SAVED_MODELS_PATH):
        super(LsaGensimModel, self).save(base_name, path)

    @classmethod
    def _load_gensim_model(cls, path):
        """
        Loads the gensim.models.LsiModel in the specified path and returns it.
        :param path: Path of the saved gensim.models.LsiModel.
        :return: The gensim.models.LsiModel.
        """
        return gensim.models.LsiModel.load(get_abspath(__file__, path))


class TopicsModelsList(metaclass=abc.ABCMeta):
    """Base class for a list of topics models."""

    _SAVE_MODELS_PATH = '../saved-models/topics/'  # Path where the models will be saved

    def __init__(self, documents):
        self.documents = documents
        self.dictionary, self.corpus = prepare_corpus(documents)
        self.models_list = []  # Stores the models created

    def create_models_and_compute_coherence_values(self, start=2, stop=20, step=1, coherence='c_v', print_and_plot=True,
                                                   title="Topic's model coherence comparison", save_plot=False,
                                                   save_plot_path=None, **kwargs):
        """
        Creates, stores and returns topics models and it's coherence values.
        Can be used to determine an optimum number of topics.
        :param start: Number of topics to start looking for the optimum.
        :param stop: Maximum number of topics to be tried.
        :param step: Number of topics to be incremented while looking for the optimum.
        :param coherence: String that represents the type of coherence to calculate.
        Valid values are: ‘c_v’, ‘c_uci’ and ‘c_npmi’.
        :param print_and_plot: If true, prints and plots the results.
        :param title: Title of the result's plot. Ignored if print_and_plot is False.
        :param save_plot: If is true and print_and_plot is True, save the plot to disk.
        :param save_plot_path: If save_plot is True and print_and_plot is True, this is the path where
        the plot will be saved.
        :param kwargs: Other keyword parameters for creating the models.
        :return: List of the created models and their corresponding coherence values.
        """
        first_index = len(self.models_list)
        current_index = first_index
        coherence_values = []

        for num_topics in tqdm(range(start, stop + 1, step)):
            # Template design pattern. create_model() is an abstract method redefined in the subclasses.
            model = self._create_model(num_topics=num_topics, **kwargs)
            # Compute coherence value
            coherence_values.append(model.compute_coherence_value(coherence))

            self.models_list.append(model)
            current_index += 1

        if print_and_plot:
            self.print_and_plot_coherence_values(self.models_list[first_index:],
                                                 coherence_values, title,
                                                 save_plot=save_plot, save_plot_path=save_plot_path)

        return self.models_list[first_index:], coherence_values

    @abc.abstractmethod
    def _create_model(self, num_topics, **kwargs):
        """
        Creates a topics model.
        :param num_topics: Number of topics of the model.
        :return: The topics model created.
        """

    def print_and_plot_coherence_values(self, models_list=None, coherence_values=None,
                                        title="Topic's model coherence comparison",
                                        save_plot=False, save_plot_path=None):
        """
        Prints and plots coherence values of the specified models.
        :param models_list: List of models. If is None, self.models_list is used.
        :param coherence_values: List of coherence values. If is None, self.coherence_values is used.
        :param title: Title of the plot.
        :param save_plot: If is True, save the plot to disk.
        :param save_plot_path: If save_plot is True, this is the path where the plot will be saved.
        Must end in '.png' or '.pdf'.
        """
        if models_list is None:
            models_list = self.models_list
        if coherence_values is None:
            coherence_values = list(map(lambda model: model.coherence_value, self.models_list))

        num_topics_list = list(map(lambda model: model.num_topics, models_list))

        # Print the coherence scores
        for num_topics, coherence_value in zip(num_topics_list, coherence_values):
            print("Num Topics =", num_topics, " has Coherence Value of", round(coherence_value, 4))

        # Plot the coherence scores
        plt.plot(np.array(num_topics_list), np.array(coherence_values))
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence score")
        plt.legend("coherence_values", loc='best')
        plt.title(title)

        # Save to disk
        if save_plot:
            plt.savefig(save_plot_path)

        plt.show()

    def save(self, base_name, path=_SAVE_MODELS_PATH, index=None):
        """
        If index parameter is None, saves all the models to disk.
        If is a number, saves only the model with that index.
        :param base_name: Base name of the models. After it, the current time, the number of topics, and
        the coherence value are added.
        :param path: Path were the models will be stored.
        :param index: Index of the model to be saved. If is none, saves all models.
        and the coherence value of the models is added.
        """
        if index is not None:
            self.models_list[index].save(base_name, path)
        else:
            for model in self.models_list:
                model.save(base_name, path)


class LdaMalletModelsList(TopicsModelsList):

    def __init__(self, documents):
        super().__init__(documents)

    def _create_model(self, num_topics, **kwargs):
        return LdaMalletModel(self.documents, self.dictionary, self.corpus, num_topics, **kwargs)

    def create_models_and_compute_coherence_values(self, start=2, stop=20, step=1, coherence='c_v', print_and_plot=True,
                                                   title="Topic's model coherence comparison", save_plot=False,
                                                   save_plot_path=None, models_base_name='mallet_model', **kwargs):
        """
        Creates, stores and returns topics models and it's coherence values.
        Can be used to determine an optimum number of topics.
        :param start: Number of topics to start looking for the optimum.
        :param stop: Maximum number of topics to be tried.
        :param step: Number of topics to be incremented while looking for the optimum.
        :param coherence: String that represents the type of coherence to calculate.
        Valid values are: ‘c_v’, ‘c_uci’ and ‘c_npmi’.
        :param print_and_plot: If true, prints and plots the results.
        :param title: Title of the result's plot. Ignored if print_and_plot is False.
        :param save_plot: If is true and print_and_plot is True, save the plot to disk.
        :param save_plot_path: If save_plot is True and print_and_plot is True, this is the path where
        the plot will be saved.
        :param models_base_name: Base name for the Mallet models. This param + num_topics of each model
        is passed to the LdaMalletModel __init__ method, because is needed for storing the files of the model.
        :param kwargs: Other keyword parameters for creating the models.
        :return: List of the created models and their corresponding coherence values.
        """
        first_index = len(self.models_list)
        current_index = first_index
        coherence_values = []

        for num_topics in tqdm(range(start, stop + 1, step)):
            # Template design pattern. create_model() is an abstract method redefined in the subclasses.
            # model_name is passed as a kwarg. This param is the reason for overrading this method.
            model = self._create_model(num_topics=num_topics, model_name=models_base_name + str(num_topics), **kwargs)
            # Compute coherence value
            coherence_values.append(model.compute_coherence_value(coherence))

            self.models_list.append(model)
            current_index += 1

        if print_and_plot:
            self.print_and_plot_coherence_values(self.models_list[first_index:],
                                                 coherence_values, title,
                                                 save_plot=save_plot, save_plot_path=save_plot_path)

        return self.models_list[first_index:], coherence_values

    # noinspection PyMethodOverriding
    def save(self, index=None):
        """
        If index parameter is None, saves all the models to disk.
        If is a number, saves only the model with that index.
        :param index: Index of the model to be saved. If is none, saves all models.
        and the coherence value of the models is added.
        """
        if index is not None:
            self.models_list[index].save()
        else:
            for model in self.models_list:
                model.save()


class LdaModelsList(TopicsModelsList):

    def __init__(self, documents):
        super().__init__(documents)

    def _create_model(self, num_topics, random_state=RANDOM_STATE, **kwargs):
        return LdaGensimModel(self.documents, self.dictionary, self.corpus, num_topics, None, random_state, **kwargs)


class LsaModelsList(TopicsModelsList):

    def __init__(self, documents):
        super().__init__(documents)

    def _create_model(self, num_topics, **kwargs):
        return LsaGensimModel(self.documents, self.dictionary, self.corpus, num_topics, **kwargs)
