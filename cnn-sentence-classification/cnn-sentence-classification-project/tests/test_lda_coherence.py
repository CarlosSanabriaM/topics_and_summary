import datetime
import os

import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.callbacks import CoherenceMetric
from tqdm import tqdm

from datasets.twenty_news_groups import TwentyNewsGroupsDataset
from preprocessing.dataset import preprocess_dataset
from utils import pretty_print, RANDOM_STATE, get_abspath_from_project_root, join_paths


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


def prepare_corpus(documents):  # TODO: Implementation is repeted from test_lsa.py
    """
    Given a list of documents, returns a term dictionary and a document-term matrix.
    :param documents: List of documents. Each document is a list of words, where each word is a string.
    :return: 2 things: Term's dictionary (it's attribute token2id is a dict mapping words to numeric identifiers)
    and Document-term matrix (a list, where each element is a list of tuples. Each tuple contains the index of a word
    and the number of times that word appears in that document).
    """
    # Creating the term dictionary of our courpus, where every unique term is assigned an index.
    dictionary = Dictionary(documents)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = get_corpus(dictionary, documents)

    return dictionary, doc_term_matrix


def print_topics(lda_model, num_words_each_topic, num_topics=None):  # TODO: Implementation is repeted from test_lsa.py
    """
    Prints the topics of the given model.
    :param lda_model:
    :param num_words_each_topic:
    :param num_topics: If not specified, default is the num topics in the lsa model.
    :return:
    """
    if num_topics is None:
        num_topics = lda_model.num_topics

    # Sequence with (topic_id, [(word, value), ... ]).
    topics_sequence = lda_model.print_topics(num_topics=num_topics, num_words=num_words_each_topic)
    for topic in topics_sequence:
        print('Topic ' + str(topic[0]) + ': ' + topic[1])


def compute_coherence_value(model, documents, terms_dictionary, coherence='c_v'):
    """
    Calculates and returns the coherence value of the given topics model.
    :param model: Topic's model
    :param documents: List of documents. Each document is a list of words, where each word is a string.
    :param terms_dictionary: Term dictionary.
    :param coherence: String that represents the type of coherence to calculate.
    Valid values are: ‘c_v’, ‘c_uci’ and ‘c_npmi’.
    :return: The coherence value of the topic model.
    """
    # Create the coherence model and evaluate the given model
    coherence_model = CoherenceModel(model=model, texts=documents, dictionary=terms_dictionary,
                                     coherence=coherence)
    return coherence_model.get_coherence()


def compute_perplexity(lda_model, doc_term_matrix):
    """
    # Measure of how good the model is. lower the better.
    :param doc_term_matrix: Document-term matrix.
    :return: Perplexity value.
    """
    return lda_model.log_perplexity(doc_term_matrix)


def compute_coherence_values(terms_dictionary, doc_term_matrix, documents, stop, start=2, step=3,
                             coherence='c_v'):  # TODO: Implementation is repeted from test_lsa.py
    """
    Generates coherence values to determine an optimum number of topics to be used to generate the given topics model.
    :param terms_dictionary: Terms dictionary.
    :param doc_term_matrix: Document-term matrix.
    :param documents: List of documents. Each document is a list of words, where each word is a string.
    :param stop: Maximum number of topics to be tried.
    :param start: Number of topics to start looking for the optimum.
    :param step: Number of topics to be incremented while looking for the optimum.
    :param coherence: String that represents the type of coherence to calculate.
    Valid values are: ‘c_v’, ‘c_uci’ and ‘c_npmi’.
    :return: A list of LDA models and their corresponding Coherence values.
    """
    coherence_values = []
    lda_models_list = []
    for num_topics in tqdm(range(start, stop, step)):
        # generate LDA model
        lda_model = LdaModel(doc_term_matrix, num_topics=num_topics, id2word=terms_dictionary)
        lda_models_list.append(lda_model)
        # Compute coherence value
        coherence_values.append(compute_coherence_value(lda_model, documents, terms_dictionary, coherence))
    return lda_models_list, coherence_values


def plot_coherence_score_values(coherence_values, start, stop,
                                step):  # TODO: Implementation is repeted from test_lsa.py
    # Plot those values
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.show()


"""LDA is easily the most popular (and typically most effective) topic modeling technique out there"""
"""Using LDA we can extract human-interpretable topics from a document corpus, where each topic is characterized 
by the words they are most strongly associated with. For example, topic 2 could be characterized by terms such as 
oil, gas, drilling, pipes, Keystone, energy, etc. 
Furthermore, given a new document, we can obtain a vector representing its topic mixture, e.g. 5% topic 1, 
70% topic 2, 10% topic 3, etc. These vectors are often very useful for downstream applications.
"""

class EpochSaver(CallbackAny2Vec):
    "Callback to save model after every epoch"
    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0
    def on_epoch_end(self, model):
        output_path = '{}_epoch{}.model'.format(self.path_prefix, self.epoch)
        print("Save model to {}".format(output_path))
        model.save(output_path)
        self.epoch += 1


class EpochLogger(CallbackAny2Vec):
    "Callback to log information about training"
    def __init__(self):
        self.epoch = 0
    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))
    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


if __name__ == '__main__':
    # Ignore deprecation warnings
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Log gensim

    # logging.basicConfig(filename=get_abspath_from_project_root('logs/lda.log'),
    #                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # %%
    # Load dataset and apply preprocessing
    dataset = TwentyNewsGroupsDataset()
    dataset = preprocess_dataset(dataset)
    # dataset, bigram_model_func = preprocess_dataset(dataset, ngrams='bi')

    # %%
    # Create the document-term matrix and the dictionary of terms
    pretty_print('Creating the document-term matrix and the dictionary of terms')
    documents = dataset.as_documents_content_list()
    terms_dictionary, doc_term_matrix = prepare_corpus(documents)

    # %%
    # Create a LDA model with 20 topics (same as num categorys) and try tu tune hyperparameteres
    pretty_print('Tuning a LDA model with 20 topics')

    # Callbacks that are called during training
    now = str(datetime.datetime.now())
    model_name = "lda_tuned_model_" + now
    os.mkdir('../saved-models/lda/' + model_name)
    path = get_abspath_from_project_root(join_paths("saved-models/lda/", model_name, model_name))

    epoch_saver = EpochSaver(path)
    epoch_logger = EpochLogger()

    #coherence_metric = CoherenceMetric(texts=documents, dictionary=terms_dictionary,
    #                                   coherence='c_v', logger='visdom', title='LDA - Coherence during training')
    #
    coherence_metric = CoherenceMetric(texts=documents, dictionary=terms_dictionary,
                                       coherence='c_v', logger='shell')

    lda_model_tuned = LdaModel(corpus=doc_term_matrix,
                               id2word=terms_dictionary,
                               num_topics=20,
                               random_state=RANDOM_STATE,
                               update_every=1,
                               chunksize=100,
                               passes=10,
                               alpha='auto',
                               per_word_topics=True,
                               callbacks=[coherence_metric])
                               #callbacks=[epoch_logger, epoch_saver, coherence_metric]) # TODO: I think that coherence_metric cant be used with the other 2

    NUM_WORDS_EACH_TOPIC_TO_BE_PRINTED = 15
    print_topics(lda_model_tuned, NUM_WORDS_EACH_TOPIC_TO_BE_PRINTED)
