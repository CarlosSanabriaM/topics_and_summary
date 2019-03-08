from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
import datetime
import os
import pyLDAvis.gensim
import matplotlib.pyplot as plt

from datasets.twenty_news_groups import TwentyNewsGroupsDataset
from preprocessing.dataset import preprocess_dataset
from utils import pretty_print, get_abspath, RANDOM_STATE


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
    Prints the topics of the given lda_model.
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


def compute_coherence_values(terms_dictionary, doc_term_matrix, documents, stop, start=2, step=3, coherence='c_v'):  # TODO: Implementation is repeted from test_lsa.py
    """
    Generates coherence values to determine an optimum number of topics to be used to generate the LDA model.
    :param terms_dictionary: Term dictionary.
    :param doc_term_matrix: Document-term matrix.
    :param documents: List of documents. Each document is a list of words, where each word is a string.
    :param stop: Maximum number of topics to be tried.
    :param start: Number of topics to start looking for the optimum.
    :param step: Number of topics to be incremented while looking for the optimum.
    :return: A list of LDA models and their corresponding Coherence values.
    """
    coherence_values = []
    lda_models_list = []
    for num_topics in range(start, stop, step):
        # generate LDA model
        lda_model = LdaModel(doc_term_matrix, num_topics=num_topics, id2word=terms_dictionary)
        lda_models_list.append(lda_model)
        # Create the coherence model and evaluate the LDA model
        coherence_model = CoherenceModel(model=lda_model, texts=documents, dictionary=terms_dictionary, coherence=coherence)
        coherence_values.append(coherence_model.get_coherence())
    return lda_models_list, coherence_values


def plot_coherence_score_values(coherence_values, start, stop, step):  # TODO: Implementation is repeted from test_lsa.py
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

if __name__ == '__main__':
    # TODO: Refactor

    # %%
    # Load dataset and apply preprocessing
    dataset = TwentyNewsGroupsDataset()
    dataset = preprocess_dataset(dataset)

    # %%
    # Create the document-term matrix and the dictionary of terms
    pretty_print('Creating the document-term matrix and the dictionary of terms')
    documents = dataset.as_documents_list()
    terms_dictionary, doc_term_matrix = prepare_corpus(documents)

    #%%
    # Train the model on the corpus.
    pretty_print('Creating the LDA model')
    lda_model = LdaModel(doc_term_matrix, num_topics=10, id2word=terms_dictionary, random_state=RANDOM_STATE)

    #%%
    # Print the most significant topics
    pretty_print('Most significant topics')
    NUM_WORDS_EACH_TOPIC_TO_BE_PRINTED = 10
    print_topics(lda_model, NUM_WORDS_EACH_TOPIC_TO_BE_PRINTED)

    #%%
    # Compute Coherence Score using c_v
    coherence_model_lda_c_v = CoherenceModel(model=lda_model, texts=documents, dictionary=terms_dictionary, coherence='c_v')
    coherence_lda_c_v = coherence_model_lda_c_v.get_coherence()
    print('\nCoherence Score: ', coherence_lda_c_v)

    #%%
    # Save model to disk.
    now = str(datetime.datetime.now())
    model_name = "lda_model_" + now

    os.mkdir('../saved-models/lda/' + model_name)
    path = get_abspath(__file__, "../saved-models/lda/" + model_name + "/" + model_name)
    lda_model.save(path)

    #%%
    # Query the model using new, unseen documents
    # TODO: Docs have to be preprocessed first
    test_docs = [
        ["Windows", "DOS", "hardware", "issue"],
        ["Trump", "guns", "U.S.A.", "problems"],
        ["Jesus", "god", "christian"],
        ["Citroen", "wheels"]
    ]
    test_corpus = get_corpus(terms_dictionary, test_docs)

    topic_prob_vector_test_doc_0 = lda_model[test_corpus[0]]  # get topic probability distribution for a document
    topic_prob_vector_test_doc_1 = lda_model[test_corpus[1]]  # get topic probability distribution for a document
    topic_prob_vector_test_doc_2 = lda_model[test_corpus[2]]  # get topic probability distribution for a document
    topic_prob_vector_test_doc_3 = lda_model[test_corpus[3]]  # get topic probability distribution for a document

    #test_corpus[0]
    #terms_dictionary.id2token[1852]
    #topic_prob_vector_test_doc_0
    #lda_model.print_topic(7)

    #%%
    # Update the model by incrementally training on the new corpus (Online training)
    #lda_model.update(test_corpus)
    #topic_prob_vector_test_doc_1_updated = lda_model[test_corpus[1]]

    #%%
    # Load model from disk
    #lda_model_from_disk = LdaModel.load(path)


    # lda_model.top_topics(...)

    #%% Visualize topics with pyLDAvis
    lda_vis_data = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, terms_dictionary)
    pyLDAvis.display(lda_vis_data)


"""
per_word_topics (bool) – If True, the model also computes a list of topics, 
sorted in descending order of most likely topics for each word, along with 
their phi values multiplied by the feature length (i.e. word count).
"""