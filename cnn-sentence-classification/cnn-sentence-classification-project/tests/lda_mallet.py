from datasets.twenty_news_groups import TwentyNewsGroupsDataset
from preprocessing.dataset import preprocess_dataset
from models.topics import LdaMalletModel
from utils import pretty_print


# def get_corpus(dictionary, documents):
#     """
#     Returns a corpus/document-term matrix, that consists on a list, where each element is a list of tuples.
#     Each list of tuples represents a document, and each tuple contains the index of a word in that document
#     and the number of times that word appears in that document.
#     :param dictionary: gensim.corpora.Dictionary object.
#     :param documents: List of lists of strings. Each one of the nested lists represents a document, and the
#     strings the words in that document.
#     :return: Returns a corpus/document-term matrix, that consists on a list, where each element is a list of tuples.
#     """
#     return [dictionary.doc2bow(doc) for doc in documents]
#
#
# def prepare_corpus(documents):  # TODO: Implementation is repeted from test_lsa.py
#     """
#     Given a list of documents, returns a term dictionary and a document-term matrix.
#     :param documents: List of documents. Each document is a list of words, where each word is a string.
#     :return: 2 things: Term's dictionary (it's attribute token2id is a dict mapping words to numeric identifiers)
#     and Document-term matrix (a list, where each element is a list of tuples. Each tuple contains the index of a word
#     and the number of times that word appears in that document).
#     """
#     # Creating the term dictionary of our courpus, where every unique term is assigned an index.
#     dictionary = Dictionary(documents)
#     # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
#     doc_term_matrix = get_corpus(dictionary, documents)
#
#     return dictionary, doc_term_matrix


# def compute_coherence_value(model, documents, terms_dictionary, coherence='c_v'):
#     """
#     Calculates and returns the coherence value of the given topics model.
#     :param model: Topic's model
#     :param documents: List of documents. Each document is a list of words, where each word is a string.
#     :param terms_dictionary: Term dictionary.
#     :param coherence: String that represents the type of coherence to calculate.
#     Valid values are: ‘c_v’, ‘c_uci’ and ‘c_npmi’.
#     :return: The coherence value of the topic model.
#     """
#     # Create the coherence model and evaluate the given model
#     coherence_model = CoherenceModel(model=model, texts=documents, dictionary=terms_dictionary,
#                                      coherence=coherence)
#     return coherence_model.get_coherence()


# def compute_coherence_values(terms_dictionary, doc_term_matrix, documents, stop, start=2, step=3,
#                              coherence='c_v'):  # TODO: Implementation is repeted from test_lsa.py
#     """
#     Generates coherence values to determine an optimum number of topics to be used to generate the given topics model.
#     :param terms_dictionary: Terms dictionary.
#     :param doc_term_matrix: Document-term matrix.
#     :param documents: List of documents. Each document is a list of words, where each word is a string.
#     :param stop: Maximum number of topics to be tried.
#     :param start: Number of topics to start looking for the optimum.
#     :param step: Number of topics to be incremented while looking for the optimum.
#     :param coherence: String that represents the type of coherence to calculate.
#     Valid values are: ‘c_v’, ‘c_uci’ and ‘c_npmi’.
#     :return: A list of LDA models and their corresponding Coherence values.
#     """
#     coherence_values = []
#     lda_models_list = []
#     for num_topics in tqdm(range(start, stop, step)):
#         # generate LDA model
#         lda_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=doc_term_matrix, num_topics=num_topics,
#                                                      id2word=terms_dictionary)
#         lda_models_list.append(lda_model)
#         # Compute coherence value
#         coherence_values.append(compute_coherence_value(lda_model, documents, terms_dictionary, coherence))
#     return lda_models_list, coherence_values


# def print_topics(lda_model, num_words_each_topic, num_topics=None):  # TODO: Implementation is repeted from test_lsa.py
#     """
#     Prints the topics of the given lda_model.
#     :param lda_model:
#     :param num_words_each_topic:
#     :param num_topics: If not specified, default is the num topics in the lsa model.
#     :return:
#     """
#     if num_topics is None:
#         num_topics = lda_model.num_topics
#
#     # Sequence with (topic_id, [(word, value), ... ]).
#     topics_sequence = lda_model.print_topics(num_topics=num_topics, num_words=num_words_each_topic)
#     for topic in topics_sequence:
#         print('Topic ' + str(topic[0]) + ': ' + topic[1])


# def plot_and_print_coherence_score_values(coherence_values, start, stop,
#                                           step):  # TODO: Implementation is repeted from test_lsa.py
#     # Plot those values
#     x = range(start, stop, step)
#     plt.plot(x, coherence_values)
#     plt.xlabel("Number of Topics")
#     plt.ylabel("Coherence score")
#     plt.legend("coherence_values", loc='best')
#     plt.show()
#
#     # Print the coherence scores
#     for m, cv in zip(x, coherence_values):
#         print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


if __name__ == '__main__':

    # %%
    # Load dataset and apply preprocessing
    dataset = TwentyNewsGroupsDataset()
    dataset = preprocess_dataset(dataset)

    #%%
    # Create the Lda Mallet model
    pretty_print('Creating the Lda Mallet model')
    documents = dataset.as_documents_list()
    model = LdaMalletModel(documents, num_topics=20)

    #%%
    # Print topics and coherence score
    pretty_print('\nTopics')
    NUM_WORDS_EACH_TOPIC_TO_BE_PRINTED = 15
    model.print_topics(NUM_WORDS_EACH_TOPIC_TO_BE_PRINTED)
    coherence_score = model.compute_coherence_value()
    pretty_print('Coherence Score')
    print(coherence_score)

    # %%
    # Save model to disk.
    model.save('lda_mallet')




    # %%
    # Create the document-term matrix and the dictionary of terms
    # pretty_print('Creating the document-term matrix and the dictionary of terms')
    # documents = dataset.as_documents_list()
    # terms_dictionary, doc_term_matrix = prepare_corpus(documents)

    # %%
    # Train the model on the corpus.
    # pretty_print('Creating the LDA Mallet model')
    # mallet_path = '../../../mallet-2.0.8/bin/mallet'
    # lda_mallet_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=doc_term_matrix, num_topics=20,
    #                                                     id2word=terms_dictionary)

    #%%
    # Print topics and coherence score
    # pretty_print('\nTopics')
    # NUM_WORDS_EACH_TOPIC_TO_BE_PRINTED = 15
    # print_topics(lda_mallet_model, NUM_WORDS_EACH_TOPIC_TO_BE_PRINTED)
    # coherence_score = compute_coherence_value(lda_mallet_model, documents, terms_dictionary)
    # print('\nCoherence Score: ', coherence_score)

    #%%
    # Save model to disk.
    # now = str(datetime.datetime.now())
    # model_name = "lda_mallet_model_" + now + ' coherence_' + str(coherence_score)
    #
    # os.mkdir('../saved-models/lda/' + model_name)
    # path = get_abspath(__file__, "../saved-models/lda/" + model_name + "/" + model_name)
    # lda_mallet_model.save(path)




    #%%
    # Compute and plot coherence values to check which number of topics is the optimum
    pretty_print('Computing the coherence values')
    START, STOP, STEP = 8, 30, 1
    lda_models_list, coherence_values = compute_coherence_values(terms_dictionary, doc_term_matrix, documents,
                                                                 start=START, stop=STOP, step=STEP)
    plot_and_print_coherence_score_values(coherence_values, START, STOP, STEP)

    # %%
    # Select the best LDA model
    pretty_print('Selecting the final LDA Mallet model')
    index_max_coherence_value = coherence_values.index(max(coherence_values))
    lda_best_mallet_model = lda_models_list[index_max_coherence_value]
    print_topics(lda_best_mallet_model, NUM_WORDS_EACH_TOPIC_TO_BE_PRINTED)

    # TODO: 18. Finding the dominant topic in each sentence ...
