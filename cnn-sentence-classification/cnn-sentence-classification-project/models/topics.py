import os
import abc
from tqdm import tqdm
import datetime
import gensim
import matplotlib.pyplot as plt
import pandas as pd

from utils import get_abspath, RANDOM_STATE


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
        :param base_name: Base name of the models. After it, the current time, the number of topics, and
        the coherence value are added.
        :param path: Path were the models will be stored.
        and the coherence value of the models is added.
        """
        if self.coherence_value is None:
            self.compute_coherence_value()

        now = str(datetime.datetime.now())
        model_name = base_name + "_" + now \
                     + " topics_" + str(self.model.num_topics) \
                     + " coherence_" + str(self.coherence_value)

        os.mkdir(path + model_name)
        path = get_abspath(__file__, path + model_name + "/" + model_name)
        self.model.save(path)

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
        model = cls._load_gensim_model(model_dir_path + model_name + "/" + model_name)
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

    def get_k_dominant_topics_of_document(self, document_index, num_best_topics=3, num_best_keywords=10):
        document = self.corpus[document_index]  # Document as a term_id-frequency list
        document_topics_contrib = self.model[document]  # List of tuples (topic_index, document_topic_contribution)

        # Order the topics by the contribution percentage of the topic in that document
        sorted_document_topics_contrib = sorted(document_topics_contrib, key=lambda x: (x[1]), reverse=True)

        dominant_topics = []
        for i in range(num_best_topics):
            topic_index, contrib = sorted_document_topics_contrib[i]
            # List of tuples (keyword_name, contribution_of_that_keyword_to_the_topic)
            topic_kewywords = self.model.show_topic(topic_index, num_best_keywords)
            dominant_topics.append((topic_index, contrib, topic_kewywords))

        return dominant_topics

    # TODO: This method hasn't been tested
    def get_dominant_topic_of_document_each_doc_as_df(self):
        """
        Returns a pandas DataFrame with the following columns: Doc_Index, Dominant_Topic_Index, Topic_Contribution,
        Topic_Keywords and Doc_Text. This method can take to much time to execute if the dataset is big.
        :return: pandas DataFrame.
        """
        docs_topics_df = pd.DataFrame(
            columns=['Doc_Index', 'Dominant_Topic_Index', 'Topic_Contribution', 'Topic_Keywords'])

        for doc_index in tqdm(range(len(self.documents))):
            dominant_topic = self.get_k_dominant_topics_of_document(doc_index, num_best_topics=1)[0]
            docs_topics_df = docs_topics_df.append(
                pd.Series([doc_index, dominant_topic[0], dominant_topic[1], dominant_topic[2]]), ignore_index=True)

        # Add original text of the documents to the end of the output
        docs_text = pd.Series(self.documents)
        docs_topics_df = pd.concat([docs_topics_df, docs_text], axis=1, names=['Doc_Text'])

        return docs_topics_df

    # TODO: This method hasn't been tested
    def get_most_representative_doc_for_each_topic_as_df(self, docs_topics_df=None):
        """
        Returns a DataFrame where each row contains a topic and the most representative document of that topic.
        :param docs_topics_df: DataFrame previously created with the method
        get_dominant_topic_of_document_each_doc_as_df(). If is None, that method is call, and that can take be slow.
        :return: A pandas DataFrame with the following columns: Topic_Index, Topic_Contribution, Topic_Keywords and
        Doc_Text.
        """
        if docs_topics_df is None:
            docs_topics_df = self.get_dominant_topic_of_document_each_doc_as_df()

        most_repr_doc_each_topic_df = pd.DataFrame()

        doc_topics_grouped_by_topic_df = docs_topics_df.groupby('Dominant_Topic_Index')

        for i, group in doc_topics_grouped_by_topic_df:
            most_repr_doc_each_topic_df = pd.concat([most_repr_doc_each_topic_df,
                                                     group.sort_values(['Topic_Contribution'], ascending=[0]).head(1)],
                                                    axis=0)

        most_repr_doc_each_topic_df.reset_index(drop=True, inplace=True)
        most_repr_doc_each_topic_df.columns = ['Topic_Index', "Topic_Contribution", "Topic_Keywords", "Doc_Text"]

        return most_repr_doc_each_topic_df

    # TODO: This method hasn't been tested
    def get_topic_distribution_as_df(self, docs_topics_df=None):
        """
        Returns a DataFrame where each row contains a topic, the number of documents of that topic (the topic
        is the dominant topic of those documents), and the percentage of documents of that topic.
        :param docs_topics_df: DataFrame previously created with the method
        get_dominant_topic_of_document_each_doc_as_df(). If is None, that method is call, and that can take be slow.
        :return: A pandas DataFrame with the following columns: Dominant_Topic_Index, Topic_Keywords, Num_Documents and
        Percentage_Documents.
        """
        if docs_topics_df is None:
            docs_topics_df = self.get_dominant_topic_of_document_each_doc_as_df()

        # Number of Documents for Each Topic
        topic_counts = docs_topics_df['Dominant_Topic_Index'].value_counts()

        # Percentage of Documents for Each Topic
        topic_contribution = round(topic_counts / topic_counts.sum(), 4)

        # Topic Number and Keywords
        topic_num_keywords = docs_topics_df[['Dominant_Topic_Index', 'Topic_Keywords']]

        # Concatenate Column wise
        df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

        # Change Column names
        df_dominant_topics.columns = ['Dominant_Topic_Index', 'Topic_Keywords', 'Num_Documents', 'Percentage_Documents']

        return df_dominant_topics


class LdaMalletModel(TopicsModel):
    """Class that encapsulates the functionality of gensim.models.wrappers.LdaMallet, making it easier to use."""

    __MALLET_SOURCE_CODE_PATH = '../../../mallet-2.0.8/bin/mallet'
    __MALLET_SAVED_MODELS_PATH = '../saved-models/topics/lda_mallet/'

    def __init__(self, documents, dictionary=None, corpus=None, num_topics=20,
                 model=None, mallet_path=get_abspath(__file__, __MALLET_SOURCE_CODE_PATH),
                 model_name=None, **kwargs):
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
        inside __MALLET_SAVED_MODELS_PATH directory. This param is obligatory.
        :param kwargs: Additional keyword arguments that want to be used in the gensim.models.wrappers.LdaMallet
        __init__ method.
        """
        if model_name is None:
            raise ValueError('model_name parameter is obligatory')

        self.model_name = model_name

        super().__init__(documents, dictionary, corpus, num_topics, model, mallet_path=mallet_path, **kwargs)

    def _create_model(self, **kwargs):
        """
        Creates the LdaMallet model and returns it.
        :param kwargs: Keyword arguments that want to be used in the gensim.models.wrappers.LdaMallet.
        mallet_path argument is obligatory.
        :return: The gensim.models.wrappers.LdaMallet model created.
        """
        # Create the folder where the mallet files will be stored
        prefix = get_abspath(__file__, self.__MALLET_SAVED_MODELS_PATH + self.model_name)
        os.mkdir(prefix)

        # Add the model_name again to the prefix. Mallet expects the prexis to be the name of a file
        # inside the folder where the mallet files will be saved. For that reason, we add to the folder
        # path the model name, and we will use later the prefix to store the final model.
        prefix += '/' + self.model_name

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

    def save(self):
        """
        Saves the mallet model to the path specified in self.model.prefix.
        """
        self.model.save(self.model.prefix)

    @classmethod
    def load(cls, model_name, documents, model_dir_path=__MALLET_SAVED_MODELS_PATH):
        return super(LdaMalletModel, cls).load(model_name, documents, model_dir_path)


class LdaModel(TopicsModel):
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

        # model_name kwargs can't be passed to LdaModel __init__()
        # TODO: This can be done better removing from kwargs all the keys that aren't in the __init__() method below
        del kwargs['model_name']

        return gensim.models.LdaModel(corpus=self.corpus,
                                      id2word=self.dictionary,
                                      num_topics=self.num_topics,
                                      **kwargs)  # random_state is passed here

    def save(self, base_name, path=__LDA_SAVED_MODELS_PATH):
        super(LdaModel, self).save(base_name, path)

    @classmethod
    def load(cls, model_name, documents, model_dir_path=__LDA_SAVED_MODELS_PATH):
        return super(LdaModel, cls).load(model_name, documents, model_dir_path)

    @classmethod
    def _load_gensim_model(cls, path):
        """
        Loads the gensim.models.LdaModel in the specified path and returns it.
        :param path: Path of the saved gensim.models.LdaModel.
        :return: The gensim.models.LdaModel.
        """
        return gensim.models.LdaModel.load(get_abspath(__file__, path))


class LsaModel(TopicsModel):
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

        # model_name kwargs can't be passed to LdaModel __init__()
        # TODO: This can be done better removing from kwargs all the keys that aren't in the __init__() method below
        del kwargs['model_name']

        return gensim.models.LsiModel(corpus=self.corpus,
                                      id2word=self.dictionary,
                                      num_topics=self.num_topics,
                                      **kwargs)

    @classmethod
    def load(cls, model_name, documents, model_dir_path=__LSA_SAVED_MODELS_PATH):
        return super(LsaModel, cls).load(model_name, documents, model_dir_path)

    def save(self, base_name, path=__LSA_SAVED_MODELS_PATH):
        super(LsaModel, self).save(base_name, path)

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

    __SAVE_PATH = '../saved-models/topics/'  # Path where the models will be saved

    def __init__(self, documents):
        self.documents = documents
        self.dictionary, self.corpus = prepare_corpus(documents)
        self.models_list = []  # Stores the models created

    def create_models_and_compute_coherence_values(self, start=2, stop=20, step=1, coherence='c_v', print_and_plot=True,
                                                   title="Topic's model coherence comparison", **kwargs):
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
        :param kwargs: Other keyword parameters for creating the models.
        :return: List of the created models and their corresponding coherence values.
        """
        first_index = len(self.models_list)
        current_index = first_index
        coherence_values = []

        for num_topics in tqdm(range(start, stop, step)):
            # Template design pattern. create_model() is an abstract method redefined in the subclasses.
            model = self._create_model(num_topics=num_topics, **kwargs)
            # Compute coherence value
            coherence_values.append(model.compute_coherence_value(coherence))

            self.models_list.append(model)
            current_index += 1

        if print_and_plot:
            self.print_and_plot_coherence_values(self.models_list[first_index:],
                                                 coherence_values,
                                                 title)

        return self.models_list[first_index:], coherence_values

    @abc.abstractmethod
    def _create_model(self, num_topics, **kwargs):
        """
        Creates a topics model.
        :param num_topics: Number of topics of the model.
        :return: The topics model created.
        """

    def print_and_plot_coherence_values(self, models_list=None, coherence_values=None,
                                        title="Topic's model coherence comparison"):
        """
        Prints and plots coherence values of the specified models.
        :param models_list: List of models. If is None, self.models_list is used.
        :param coherence_values: List of coherence values. If is None, self.coherence_values is used.
        :param title: Title of the plot.
        """
        if models_list is None:
            models_list = self.models_list
        if coherence_values is None:
            coherence_values = map(lambda m: m.coherence_value, self.models_list)

        num_topics_list = map(lambda model: model.num_topics, models_list)

        # Print the coherence scores
        for num_topics, coherence_value in zip(num_topics_list, coherence_values):
            print("Num Topics =", num_topics, " has Coherence Value of", round(coherence_value, 4))

        # Plot the coherence scores
        plt.plot(list(num_topics_list), coherence_values)
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence score")
        plt.legend("coherence_values", loc='best')
        plt.title(title)
        plt.show()

    def save(self, base_name, path=__SAVE_PATH, index=None):
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


class LdaModelsList(TopicsModelsList):

    def __init__(self, documents):
        super().__init__(documents)

    def _create_model(self, num_topics, random_state=RANDOM_STATE, **kwargs):
        return LdaModel(self.documents, self.dictionary, self.corpus, num_topics, random_state, **kwargs)


class LsaModelsList(TopicsModelsList):

    def __init__(self, documents):
        super().__init__(documents)

    def _create_model(self, num_topics, **kwargs):
        return LsaModel(self.documents, self.dictionary, self.corpus, num_topics, **kwargs)
