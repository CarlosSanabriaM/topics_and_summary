from gensim.corpora import Dictionary
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from datasets.twenty_news_groups import TwentyNewsGroupsDataset
from preprocessing.dataset import preprocess_dataset
from utils import pretty_print


def prepare_corpus(documents):
    """
    Given a list of documents, returns a term dictionary and a document-term matrix.
    :param documents: List of documents. Each document is a list of words, where each word is a string.
    :return: 2 things: Term dictionary (it's attribute token2id is a dict mapping words to numeric identifiers)
    and Document-term matrix (a list, where each element is a list of tuple. Each tuple contains the index of a word
    and the number of times that word appears in that document).
    """
    # Creating the term dictionary of our courpus, where every unique term is assigned an index.
    dictionary = Dictionary(documents)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in documents]

    return dictionary, doc_term_matrix


def create_gensim_lsa_model(terms_dictionary, doc_term_matrix, num_topics, num_words_each_topic):
    """
    Creates the LSA model.
    :param terms_dictionary: Term dictionary.
    :param doc_term_matrix: Document-term matrix.
    :param num_topics: Number of topics.
    :param num_words_each_topic: Number of words associated with each topic to be printed.
    :return: LSA model.
    """
    # generate LSA model (LSI is the same as LSA)
    lsa_model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word=terms_dictionary)  # train model
    print_topics(lsa_model, num_words_each_topic)
    return lsa_model


def compute_coherence_values(terms_dictionary, doc_term_matrix, documents, stop, start=2, step=3):
    """
    Generates coherence values to determine an optimum number of topics to be used to generate the LSA model.
    :param terms_dictionary: Term dictionary.
    :param doc_term_matrix: Document-term matrix.
    :param documents: List of documents. Each document is a list of words, where each word is a string.
    :param stop: Maximum number of topics to be tried.
    :param start: Number of topics to start looking for the optimum.
    :param step: Number of topics to be incremented while looking for the optimum.
    :return: A list of LSA models and their corresponding Coherence values.
    """
    coherence_values = []
    lsa_models_list = []
    for num_topics in range(start, stop, step):
        # generate LSA lda_model
        lsa_model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word=terms_dictionary)
        lsa_models_list.append(lsa_model)
        # Create the coherence model and evaluate the LSA model
        coherence_model = CoherenceModel(model=lsa_model, texts=documents, dictionary=terms_dictionary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
    return lsa_models_list, coherence_values


def plot_coherence_score_values(coherence_values, start, stop, step):
    # Plot those values
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.show()


def print_topics(lsa_model, num_words_each_topic, num_topics=None):
    """
    Prints the topics of the given lda_model.
    :param lsa_model:
    :param num_words_each_topic:
    :param num_topics: If not specified, default is the num topics in the lsa model.
    :return:
    """
    if num_topics is None:
        num_topics = lsa_model.num_topics

    # Sequence with (topic_id, [(word, value), ... ]).
    topics_sequence = lsa_model.print_topics(num_topics=num_topics, num_words=num_words_each_topic)
    for topic in topics_sequence:
        print('Topic ' + str(topic[0]) + ': ' + topic[1])


if __name__ == '__main__':
    # %%
    # Load dataset and apply preprocessing
    dataset = TwentyNewsGroupsDataset()
    dataset = preprocess_dataset(dataset)

    # %%
    # Create the document-term matrix and the dictionary of terms
    pretty_print('Creating the document-term matrix and the dictionary of terms')
    documents = dataset.as_documents_list()
    terms_dictionary, doc_term_matrix = prepare_corpus(documents)

    # %%
    # Compute and plot coherence values to check which number of topics is the optimum
    pretty_print('Computing the coherence values')
    START, STOP, STEP = 2, 12, 1
    lsa_models_list, coherence_values = compute_coherence_values(terms_dictionary, doc_term_matrix, documents,
                                                                 start=START, stop=STOP, step=STEP)

    plot_coherence_score_values(coherence_values, START, STOP, STEP)

    # %%
    # Select the best LSA model
    pretty_print('Selecting the final LSA model')
    NUM_WORDS_EACH_TOPIC_TO_BE_PRINTED = 10
    index_max_coherence_value = coherence_values.index(max(coherence_values))
    lsa_model = lsa_models_list[index_max_coherence_value]
    print_topics(lsa_model, NUM_WORDS_EACH_TOPIC_TO_BE_PRINTED)
