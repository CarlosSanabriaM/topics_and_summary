import abc
import os
from typing import List, Tuple, Callable

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from texttable import Texttable
from tqdm import tqdm

from datasets.common import Dataset
from preprocessing.text import preprocess_text
from utils import RANDOM_STATE, now_as_str, join_paths, get_abspath_from_project_root


def prepare_corpus(documents) -> Tuple[gensim.corpora.Dictionary, List[List[Tuple[int, int]]]]:
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


def get_corpus(dictionary, documents) -> List[List[Tuple[int, int]]]:
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
    """Base class that represents a topics model."""

    __SAVE_PATH = get_abspath_from_project_root('saved-elements/topics/')  # Path where the models will be saved

    def __init__(self, dataset: Dataset, dictionary: gensim.corpora.Dictionary = None,
                 corpus: List[List[Tuple[int, int]]] = None, num_topics=20,
                 model=None, docs_topics_df: pd.DataFrame = None, **kwargs):
        """
        :param dataset: Dataset.
        :param dictionary: gensim.corpora.Dictionary object. If is None, it is created using the dataset documents.
        :param corpus: Document-term matrix. If is None, it is created using the dataset documents.
        :param num_topics: Number of topics.
        :param model: Pre-created model. If is None, a model is created.
        :param docs_topics_df: DataFrame with the dominant topic of each document, previously created with the method
        get_dominant_topic_of_each_doc_as_df().
        :param kwargs: Additional arguments.
        """
        self.dataset = dataset
        self.document_objects_list = dataset.as_documents_obj_list()
        # The line below is way faster than dataset.as_documents_content_list()
        self.documents = list(map(lambda d: d.content, self.document_objects_list))

        self.num_topics = num_topics
        self.coherence_value = None
        self.docs_topics_df = docs_topics_df
        self.dir_path = None  # Path of the directory where the model is saved to

        if dictionary is None or corpus is None:
            self.dictionary, self.corpus = prepare_corpus(self.documents)
        else:
            self.dictionary, self.corpus = dictionary, corpus

        if model is None:
            self.model = self._create_model(**kwargs)
        else:
            self.model = model

    @abc.abstractmethod
    def _create_model(self, num_topics: int, **kwargs):
        """
        Factory Method design pattern. The subclasses override this method,
        creating and returning the specific model that the subclasses represent.
        :param num_topics: Number of topics of the model.
        """

    def compute_coherence_value(self, coherence='c_v') -> float:
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

    def save(self, model_name: str, path=__SAVE_PATH, add_metadata_to_base_name=False):
        """
        Saves the model to disk.
        :param model_name: Name of the model.
        :param path: Path were the models will be stored.
        :param add_metadata_to_base_name: If True, the number of topics, the coherence value and the current time
        are added at the end of the model name.
        """
        # Coherence value is calculated even if add_metadata_to_base_name is False,
        # because a file with it's value is created and stored inside the model folder.
        if self.coherence_value is None:
            self.compute_coherence_value()

        if add_metadata_to_base_name:
            now = now_as_str()
            model_name = "{0}_{1}topics_coherence{2}_{3}".format(model_name, str(self.model.num_topics),
                                                                 str(self.coherence_value), now)

        self.dir_path = join_paths(path, model_name)
        os.mkdir(self.dir_path)
        model_path = join_paths(self.dir_path, model_name)
        self.model.save(model_path)

        # Save the coherence value in a .txt file
        coherence_path = join_paths(self.dir_path, "coherence_value.txt")
        with open(coherence_path, 'w') as f:
            f.write(str(self.coherence_value))

    @classmethod
    def load(cls, model_name: str, dataset: Dataset, model_dir_path=__SAVE_PATH, docs_topics_df: pd.DataFrame = None):
        """
        Loads the model with the given name from the specified path, and
        returns a TopicsModel instance.
        :param model_name: Model name.
        :param dataset: Dataset.
        :param model_dir_path: Path to the directory where the model is in.
        :param docs_topics_df: DataFrame with the dominant topic of each document, previously created with the method
        get_dominant_topic_of_each_doc_as_df().
        :return: Instance of a TopicsModel object.
        """
        model = cls._load_gensim_model(join_paths(model_dir_path, model_name, model_name))
        return cls(dataset, num_topics=model.num_topics, model=model, model_name=model_name,
                   docs_topics_df=docs_topics_df)

    @classmethod
    @abc.abstractmethod
    def _load_gensim_model(cls, path: str):
        """
        Factory Method design pattern. The subclasses override this method,
        loading the gensim model in the specified path and returning it.
        :param path: Path of the saved gensim model.
        :return: The gensim model.
        """

    def print_topics(self, num_keywords=10, gensim_way=True):
        """
        Prints the topics of the topics model.
        :param num_keywords: Number of keywords of each topic to be printed.
        :param gensim_way: If True, the topics are printed in the gensim way.
        If not, are printed using the __str__ methods of the Topic and Keywords classes.
        """
        if gensim_way:
            # Sequence with (topic_id, [(word, value), ... ]).
            topics_sequence = self.model.print_topics(num_topics=self.num_topics, num_words=num_keywords)
            for topic in topics_sequence:
                print('Topic ' + str(topic[0]) + ': ' + topic[1])
        else:
            print(*self.get_topics(num_keywords), sep='\n')

    def get_topics(self, num_keywords=10) -> List['Topic']:
        """
        Returns a list of the topics and it's keywords (keyword name and keyword probability).
        Keywords inside a topic are ordered by it's probability inside that topic.
        :param num_keywords: Number of keywords of each topic.
        :return: List of Topics objs.
        """
        return [Topic(id, kws_as_list_of_tuples)
                for id, kws_as_list_of_tuples in
                self.model.show_topics(num_topics=self.num_topics, num_words=num_keywords, formatted=False)]

    def get_topic(self, topic: int, num_keywords=10) -> 'Topic':
        """
        Returns a list of the topic keywords (keyword name and keyword probability).
        Keywords are ordered by it's probability inside the topic.
        :param topic: Topic id.
        :param num_keywords: Number of keywords to retrieve.
        :return: Topic obj.
        """
        return Topic(topic, self.model.show_topic(topic, num_keywords))

    def predict_topic_prob_on_text(self, text: str, num_best_topics: int = None, preprocess=True,
                                   ngrams='uni', ngrams_model_func: Callable = None, print_table=True) \
            -> List[Tuple[int, float]]:
        """
        Predicts the probability of each topic to be related to the given text.
        The probabilities sum 1. When the probability of a topic is very high,
        the other topics may not appear in the results.
        :param text: Text.
        :param num_best_topics: Number of topics to return. If is None, returns all the topics that the model returns.
        :param preprocess: If true, applies preprocessing to the given text using preprocessing.text.preprocess_text().
        :param ngrams: If 'uni', uses unigrams. If 'bi', create bigrams. If 'tri', creates trigrams.
        By default is 'uni'. If is 'bi' or 'tri', it uses the ngrams_model_func for creating the bi/trigrams.
        :param ngrams_model_func: Function that receives a list of words and returns a list of words with
        possible bigrams/trigrams, based on the bigram/trigram model trained in the given dataset. This function
        is returned by make_bigrams_and_get_bigram_model_func() or make_trigrams_and_get_trigram_model_func() functions
        in the preprocessing.ngrams module. If ngrams is 'uni' this function is not used.
        :param print_table: If True, this method also prints a table with the topics indices,
        their probabilities, and their keywords.
        :return: Topic probability vector.
        """
        if preprocess:
            text = preprocess_text(text, ngrams=ngrams, ngrams_model_func=ngrams_model_func)

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

    def get_related_docs_as_df(self, text: str, num_docs=5, preprocess=True, ngrams='uni',
                               ngrams_model_func: Callable = None, remove_duplicates=True) -> pd.DataFrame:
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
        :param num_docs: Number of related documents to retrieve.
        :param preprocess: If True, apply preprocessing to the text.
        :param ngrams: If 'uni', uses unigrams. If 'bi', create bigrams. If 'tri', creates trigrams.
        By default is 'uni'. If is 'bi' or 'tri', it uses the ngrams_model_func for creating the bi/trigrams.
        :param ngrams_model_func: Function that receives a list of words and returns a list of words with
        possible bigrams/trigrams, based on the bigram/trigram model trained in the given dataset. This function
        is returned by make_bigrams_and_get_bigram_model_func() or make_trigrams_and_get_trigram_model_func() functions
        in the preprocessing.ngrams module. If ngrams is 'uni' this function is not used.
        :param remove_duplicates: If True, duplicate documents are not present in the returned DataFrame.
        Even so, num_docs documents are returned, obtained from below of the removed documents (the documents are
        ordered descending).
        :return: The pandas DataFrame.
        """
        # 1. Obtain the list of topics more related with the text
        topic_prob_vector = self.predict_topic_prob_on_text(text, preprocess=preprocess, ngrams=ngrams,
                                                            ngrams_model_func=ngrams_model_func, print_table=False)

        topics = list(map(lambda x: x[0], topic_prob_vector))  # Stores the indices of the topics

        # 2. Obtain a df with the documents more related with the topics in the previous step
        # Maybe all the best docs are from a unique topic,
        # so we need to obtain num_docs docs from each topic in the topics list
        k_most_repr_doc_per_topic_df = self.get_k_most_repr_docs_per_topic_as_df(num_docs, remove_duplicates)
        # Only the docs of the topics in the topics list are kept
        related_docs_df = k_most_repr_doc_per_topic_df.loc[k_most_repr_doc_per_topic_df['Topic index'].isin(topics)]

        # 3. Transform the df to have the following columns: Doc index, Doc prob, Doc text, Topic index, Topic keywords
        # Doc prob = prob of the text being related with the topic * prob that the doc influence the topic

        def get_text_topic_prob_of_topic(topic_index):
            return next(filter(lambda x: x[0] == topic_index, topic_prob_vector))[1]

        # Iterate for each row, get the prob text-topic from the topic_prob_vector using the 'Topic index' of that row,
        # get the prob doc-topic from that row, and multiply them. 'Topic prob' is the doc-topic prob.
        doc_prob_column = related_docs_df.apply(lambda row:
                                                get_text_topic_prob_of_topic(row['Topic index']) * row['Topic prob'],
                                                axis='columns')

        # Add the 'Doc prob' column
        related_docs_df.insert(2, 'Doc prob', doc_prob_column, allow_duplicates=True)

        # Change columns order
        related_docs_df = related_docs_df[['Doc index', 'Doc prob', 'Doc text', 'Original doc text',
                                           'Topic index', 'Topic keywords']]

        # Order by 'Doc prob' column in descending order
        related_docs_df = related_docs_df.sort_values(['Doc prob'], ascending=[False])

        if remove_duplicates:
            # Duplicate documents can appear in different topics. If that happens, here duplicates are removed,
            # keeping only the document with the highest probability.
            related_docs_df.drop_duplicates(subset='Doc text', keep='first', inplace=True)

        # Reset the indices
        related_docs_df.reset_index(drop=True, inplace=True)

        # Only the first num_docs rows are kept
        related_docs_df.drop(related_docs_df.index[range(num_docs, len(related_docs_df.index))], inplace=True)

        return related_docs_df

    def get_dominant_topic_of_each_doc_as_df(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame with the dominant topic of each document.
        The df has the following columns: Doc index, Dominant topic index, Topic prob,Topic keywords, Doc text.
        This method can take to much time to execute if the dataset is big.
        :return: pandas DataFrame.
        """
        if self.docs_topics_df is not None:
            return self.docs_topics_df

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

    def get_k_most_repr_docs_per_topic_as_df(self, k=1, remove_duplicates=True) -> pd.DataFrame:
        """
        Returns a DataFrame where the topics are grouped in ascending order by their indices, and inside each
        topic group there are k rows, where each row contains the topic and one of the most representative documents
        of that topic, in descending order.
        :param k: Number of the most representative documents per topic you want.
        :param remove_duplicates: If True, duplicate documents are not present in the same topic in the returned
        DataFrame. Even so, k documents per topic are returned, obtained from below of the removed documents
        (the documents are ordered descending).
        :return: A pandas DataFrame with the following columns: Topic index, Doc index, Topic prob, Topic keywords and
        Doc text.
        """
        if self.docs_topics_df is None:
            self.get_dominant_topic_of_each_doc_as_df()

        k_most_repr_doc_per_topic_df = pd.DataFrame()

        # Group rows by the topic index
        doc_topics_grouped_by_topic_df = self.docs_topics_df.groupby('Dominant topic index')

        # For each topic group, sort the docs by the 'Topic prob' (in descending order) and select k ones
        for topic, group in doc_topics_grouped_by_topic_df:
            topic_ordered_docs = group.sort_values(['Topic prob'], ascending=[False])

            if remove_duplicates:
                # The first k non duplicate documents are selected
                k_non_duplicate_docs_list = []
                i = 0
                while len(k_non_duplicate_docs_list) < k:
                    if topic_ordered_docs.iloc[i]['Doc text'] not in \
                            map(lambda pd_series: pd_series['Doc text'], k_non_duplicate_docs_list):
                        k_non_duplicate_docs_list.append(
                            topic_ordered_docs.iloc[i]
                        )
                    i += 1
                # Concat the series list into a single df
                most_repr_docs = pd.DataFrame(k_non_duplicate_docs_list)
            else:
                # The first k documents of each topic are selected (duplicate documents can exist)
                most_repr_docs = topic_ordered_docs.head(k)

            k_most_repr_doc_per_topic_df = pd.concat([k_most_repr_doc_per_topic_df, most_repr_docs],
                                                     axis=0)

        # Reset indices
        k_most_repr_doc_per_topic_df.reset_index(drop=True, inplace=True)
        # Change columns names
        k_most_repr_doc_per_topic_df.columns = ['Doc index', 'Topic index', 'Topic prob', 'Topic keywords', 'Doc text']
        # Change columns order
        k_most_repr_doc_per_topic_df = \
            k_most_repr_doc_per_topic_df[['Topic index', 'Doc index', 'Topic prob', 'Doc text', 'Topic keywords']]

        # Add a last column with the original document text, obtained from disk (no preprocessing, no tokenization, ...)
        #  Iterate for each row and get the value of the column 'Doc index'. Using the doc index, obtain the
        #  document object with that index from the document_objects_list. Using that document object and the dataset,
        #  obtain the document original content from disk.
        original_doc_text_column = \
            k_most_repr_doc_per_topic_df.apply(lambda row:
                                               self.dataset.get_original_doc_content_from_disk(
                                                   self.document_objects_list[row['Doc index']]
                                               ),
                                               axis='columns')

        # Add the 'Original doc text' column
        k_most_repr_doc_per_topic_df.insert(k_most_repr_doc_per_topic_df.shape[1], 'Original doc text',
                                            original_doc_text_column, allow_duplicates=True)

        return k_most_repr_doc_per_topic_df

    def get_k_most_repr_docs_of_topic_as_df(self, topic: int, k=1, remove_duplicates=True) -> pd.DataFrame:
        """
        Returns a DataFrame with the k most representative documents of the given topic.
        The DataFrame has k rows, where each row contains the document index, the document-topic probability and
        the document text, in descending order.
        :param topic: Index of the topic.
        :param k: Number of the most representative documents of the topic you want.
        :param remove_duplicates: If True, duplicate documents are not present in the returned DataFrame.
        Even so, k documents are returned, obtained from below of the removed documents (the documents are
        ordered descending).
        :return: A pandas DataFrame with the following columns: Doc index, Topic prob and Doc text.
        """
        k_most_repr_doc_per_topic_df = self.get_k_most_repr_docs_per_topic_as_df(k, remove_duplicates)

        # Keep only the rows where the 'Topic index' equals the topic index passed as a parameter
        k_most_repr_doc_per_topic_df = \
            k_most_repr_doc_per_topic_df.loc[k_most_repr_doc_per_topic_df['Topic index'] == topic]

        k_most_repr_doc_per_topic_df.reset_index(inplace=True)

        # Return a df with only the following columns: Doc index, Topic prob, Doc text and Original doc text
        return k_most_repr_doc_per_topic_df[['Doc index', 'Topic prob', 'Doc text', 'Original doc text']]

    def get_topic_distribution_as_df(self) -> pd.DataFrame:
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
        topic_kws = pd.Series([self.model.show_topic(i) for i in range(self.num_topics)])
        df_dominant_topics = pd.concat([df_dominant_topics, topic_kws], axis=1)

        # Change Column names
        df_dominant_topics.columns = ['Dominant topic index', 'Num docs', 'Percentage docs', 'Topic keywords']

        return df_dominant_topics

    def get_doc_topic_prob_matrix(self) -> np.ndarray:
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

    def get_k_kws_of_topic_as_str(self, topic: int, k: int):
        """
        :param topic: Topic where keywords will be obtained.
        :param k: Number of keywords to be returned.
        :return: k keywords from the given topic as a str.
        """
        return ' '.join(list(map(lambda x: x[0], self.model.show_topic(topic)))[:k])

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            # docs_topics_df can be None, so we need to do this if-else to check it's equality
            if self.docs_topics_df is None and other.docs_topics_df is not None:
                docs_topics_df_are_equal = False
            else:
                # Here self.docs_topics_df can't be None
                docs_topics_df_are_equal = (self.docs_topics_df is None and other.docs_topics_df is None) or \
                                           self.docs_topics_df.equals(other.docs_topics_df)

            return self.dataset == other.dataset and \
                   self.num_topics == other.num_topics and \
                   self.compute_coherence_value() == other.compute_coherence_value() and \
                   docs_topics_df_are_equal

        return False


class Topic:
    """
    Class that simply stores the id of a topic and a specific number of keywords obtained from that topic.
    It's used for data transfer.
    """

    def __init__(self, id: int, kws_as_list_of_tuples: List[Tuple[str, float]]):
        """
        :param id: Id of the topic
        :param kws_as_list_of_tuples: Keywords obtained from the topic.
        :type kws_as_list_of_tuples: List[Tuple[str, float]]
        """
        self.id = id
        self.keywords = [Keyword(name, probability) for name, probability in kws_as_list_of_tuples]

    def __str__(self):
        s = 'Topic {0}:\n' \
            '\tNum keywords={1}\n' \
            '\tKeywords:\n' \
            .format(self.id, self.num_keywords())

        for kw in self.keywords:
            s += '\t\t{0}\n'.format(kw)

        return s

    def num_keywords(self) -> int:
        """
        :return: Number of keywords stored in this topic object.
        """
        return len(self.keywords)

    def as_list_of_tuples(self) -> List[Tuple[str, float]]:
        """
        :return: Topics as a list of tuples: [(name1, prob1), (name2, prob2), ...].
        """
        return [kw.as_tuple() for kw in self.keywords]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.id == other.id and \
                   self.keywords == other.keywords
        return False


class Keyword:
    """
    Class that simply stores the name of a keywords and it's probability inside a topic.
    It's used for data transfer.
    """

    def __init__(self, name: str, probability: float):
        """
        :param name: Name of the keyword.
        :param probability: Probability of the keyword inside it's topic.
        """
        self.name = name
        self.probability = probability

    def __str__(self):
        return '{0}: {1}'.format(self.name, self.probability)

    def as_tuple(self) -> Tuple[str, float]:
        """
        :return: Keyword as a tuple: (name, prob).
        """
        return self.name, self.probability

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.name == other.name and \
                   self.probability == other.probability
        return False


class LdaMalletModel(TopicsModel):
    """Class that encapsulates the functionality of gensim.models.wrappers.LdaMallet, making it easier to use."""

    __MALLET_SOURCE_CODE_PATH = get_abspath_from_project_root('../../mallet-2.0.8/bin/mallet')
    __MALLET_SAVED_MODELS_PATH = get_abspath_from_project_root('saved-elements/topics/lda_mallet')

    def __init__(self, dataset: Dataset, dictionary: gensim.corpora.Dictionary = None,
                 corpus: List[List[Tuple[int, int]]] = None, num_topics=20, model=None,
                 mallet_path=__MALLET_SOURCE_CODE_PATH, model_name: str = None,
                 model_path=__MALLET_SAVED_MODELS_PATH, **kwargs):
        """
        Encapsulates the functionality of gensim.models.wrappers.LdaMallet, making it easier to use.
        :param dataset: Dataset.
        :param dictionary: gensim.corpora.Dictionary object. If is None, it is created using the dataset documents.
        :param corpus: Document-term matrix. If is None, it is created using the dataset documents.
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

        super().__init__(dataset, dictionary, corpus, num_topics, model,
                         mallet_path=mallet_path, model_path=model_path, **kwargs)

    def _create_model(self, **kwargs):
        """
        Creates the LdaMallet model and returns it.
        :param kwargs: Keyword arguments that want to be used in the gensim.models.wrappers.LdaMallet.
        mallet_path argument is obligatory.
        :return: The gensim.models.wrappers.LdaMallet model created.
        """
        # Create the folder where the mallet files will be stored

        prefix = join_paths(kwargs['model_path'], self.model_name)
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

    # noinspection PyMethodOverriding
    @classmethod
    def _load_gensim_model(cls, path: str, mallet_path: str):
        """
        Loads the gensim.models.wrappers.LdaMallet in the specified path and returns it.
        :param path: Path of the saved gensim.models.wrappers.LdaMallet.
        :return: The gensim.models.wrappers.LdaMallet model.
        """
        model = gensim.models.wrappers.LdaMallet.load(path)
        # Save path in the prefix attribute of the mallet model, because it's needed to access the files
        model.prefix = path
        # Save mallet path in the mallet_path attribute
        model.mallet_path = mallet_path

        return model

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
    def load(cls, model_name: str, dataset: Dataset,
             model_dir_path=__MALLET_SAVED_MODELS_PATH, mallet_path=__MALLET_SOURCE_CODE_PATH,
             docs_topics_df: pd.DataFrame = None):
        """
        Loads the model with the given name from the specified path, and
        returns a LdaMalletModel instance.
        :param model_name: Model name.
        :param dataset: Dataset used to create the model previously.
        :param model_dir_path: Path to the directory where the model is in.
        :param mallet_path: Path to the mallet source code.
        :param docs_topics_df: DataFrame with the dominant topic of each document, previously created with the method
        get_dominant_topic_of_each_doc_as_df().
        :return: Instance of a LdaMalletModel object.
        """
        model = cls._load_gensim_model(join_paths(model_dir_path, model_name, model_name), mallet_path)
        return cls(dataset, num_topics=model.num_topics, model=model, model_name=model_name,
                   docs_topics_df=docs_topics_df)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return super().__eq__(other) and self.model_name == other.model_name
        return False


class LdaGensimModel(TopicsModel):
    """Class that encapsulates the functionality of gensim.models.LdaModel, making it easier to use."""

    __LDA_SAVED_MODELS_PATH = get_abspath_from_project_root('saved-elements/topics/lda/')

    def __init__(self, dataset: Dataset, dictionary: gensim.corpora.Dictionary = None,
                 corpus: List[List[Tuple[int, int]]] = None, num_topics=20,
                 model=None, random_state=RANDOM_STATE, **kwargs):
        """
        Encapsulates the functionality of gensim.models.LdaModel, making it easier to use.
        :param dataset: Dataset.
        :param dictionary: gensim.corpora.Dictionary object. If is None, it is created using the dataset documents.
        :param corpus: Document-term matrix. If is None, it is created using the dataset documents.
        :param num_topics: Number of topics.
        :param model: Pre-created model. If is None, a model is created.
        :param random_state: Random state for reproducibility.
        :param kwargs: Additional keyword arguments that want to be used in the gensim.models.LdaModel __init__ method.
        """
        super().__init__(dataset, dictionary, corpus, num_topics, model, random_state=random_state, **kwargs)

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

    def save(self, model_name: str, path=__LDA_SAVED_MODELS_PATH, add_metadata_to_base_name=False):
        super(LdaGensimModel, self).save(model_name, path, add_metadata_to_base_name)

    @classmethod
    def load(cls, model_name: str, dataset: Dataset, model_dir_path=__LDA_SAVED_MODELS_PATH,
             docs_topics_df: pd.DataFrame = None):
        return super(LdaGensimModel, cls).load(model_name, dataset, model_dir_path, docs_topics_df)

    @classmethod
    def _load_gensim_model(cls, path: str):
        """
        Loads the gensim.models.LdaModel in the specified path and returns it.
        :param path: Path of the saved gensim.models.LdaModel.
        :return: The gensim.models.LdaModel.
        """
        return gensim.models.LdaModel.load(path)


class LsaGensimModel(TopicsModel):
    """Class that encapsulates the functionality of gensim.models.LsiModel, making it easier to use."""

    __LSA_SAVED_MODELS_PATH = get_abspath_from_project_root('saved-elements/topics/lsa/')

    def __init__(self, dataset: Dataset, dictionary: gensim.corpora.Dictionary = None,
                 corpus: List[List[Tuple[int, int]]] = None, num_topics=20, model=None, **kwargs):
        """
        Encapsulates the functionality of gensim.models.LsiModel, making it easier to use.
        :param dataset: Dataset.
        :param dictionary: gensim.corpora.Dictionary object. If is None, it is created using the dataset documents.
        :param corpus: Document-term matrix. If is None, it is created using the dataset documents.
        :param num_topics: Number of topics.
        :param model: Pre-created model. If is None, a model is created.
        :param kwargs: Additional keyword arguments that want to be used in the gensim.models.LsiModel __init__ method.
        """
        super().__init__(dataset, dictionary, corpus, num_topics, model, **kwargs)

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
    def load(cls, model_name: str, dataset: Dataset, model_dir_path=__LSA_SAVED_MODELS_PATH,
             docs_topics_df: pd.DataFrame = None):
        return super(LsaGensimModel, cls).load(model_name, dataset, model_dir_path, docs_topics_df)

    def save(self, model_name: str, path=__LSA_SAVED_MODELS_PATH, add_metadata_to_base_name=False):
        super(LsaGensimModel, self).save(model_name, path, add_metadata_to_base_name)

    @classmethod
    def _load_gensim_model(cls, path: str):
        """
        Loads the gensim.models.LsiModel in the specified path and returns it.
        :param path: Path of the saved gensim.models.LsiModel.
        :return: The gensim.models.LsiModel.
        """
        return gensim.models.LsiModel.load(path)


class TopicsModelsList(metaclass=abc.ABCMeta):
    """Base class for creating, comparing and storing easily a list of topics models."""

    _SAVE_MODELS_PATH = get_abspath_from_project_root('saved-elements/topics/')  # Path where the models will be saved

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

        # Dictionary and corpus are calculated here, to avoid recalculating them in the moment each model is created
        documents = dataset.as_documents_content_list()
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
    def _create_model(self, num_topics: int, **kwargs):
        """
        Creates a topics model.
        :param num_topics: Number of topics of the model.
        :return: The topics model created.
        """

    def print_and_plot_coherence_values(self, models_list=None, coherence_values=None,
                                        title="Topic's model coherence comparison",
                                        save_plot=False, save_plot_path: str = None):
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

    def save(self, base_name: str, path=_SAVE_MODELS_PATH, index: int = None):
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
    """Class for creating, comparing and storing easily a list of LdaMalletModels."""

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def _create_model(self, num_topics: int, **kwargs) -> LdaMalletModel:
        return LdaMalletModel(self.dataset, self.dictionary, self.corpus, num_topics, **kwargs)

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
    def save(self, index: int = None):
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
    """Class for creating, comparing and storing easily a list of LdaModels."""

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def _create_model(self, num_topics: int, random_state=RANDOM_STATE, **kwargs) -> LdaGensimModel:
        return LdaGensimModel(self.dataset, self.dictionary, self.corpus, num_topics, None, random_state, **kwargs)


class LsaModelsList(TopicsModelsList):
    """Class for creating, comparing and storing easily a list of LsaModels."""

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def _create_model(self, num_topics: int, **kwargs) -> LsaGensimModel:
        return LsaGensimModel(self.dataset, self.dictionary, self.corpus, num_topics, **kwargs)
