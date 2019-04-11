from datasets.twenty_news_groups import TwentyNewsGroupsDataset
from models.topics import LdaMalletModel, LdaMalletModelsList
from preprocessing.dataset import preprocess_dataset
from utils import pretty_print

if __name__ == '__main__':

    # %%
    # Load dataset and apply preprocessing
    dataset = TwentyNewsGroupsDataset()
    dataset = preprocess_dataset(dataset)

    #%%
    # Create the Lda Mallet model
    pretty_print('Creating the Lda Mallet model')
    documents = dataset.as_documents_content_list()
    model = LdaMalletModel(documents, num_topics=20, model_name='lda_mallet_test')

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
    model.save()

    #%%
    # Create various models and compute and plot coherence values to check which number of topics is the optimum
    pretty_print('Creating models and computing the coherence values')
    START, STOP, STEP = 8, 30, 1
    models_list = LdaMalletModelsList(documents)
    models, coherence_values = models_list.create_models_and_compute_coherence_values(
        START, STOP, STEP, title='Lda Mallet models comparison')

    # Save them
    models_list.save("lda_mallet")

    # %%
    # Select the best LDA model
    pretty_print('Selecting the final LDA Mallet model')
    index_max_coherence_value = coherence_values.index(max(coherence_values))
    best_model = models_list[index_max_coherence_value]
    best_model.print_topics(NUM_WORDS_EACH_TOPIC_TO_BE_PRINTED)

    # %%
    # Finding the dominant topics of documents
    # documents_indices = range(0, 20001, 1000)
    # for doc_index in documents_indices:
    #     pretty_print("\nDocument index: " + str(doc_index))
    #     print(documents[doc_index])
    #     pprint(best_model.get_k_dominant_topics_of_document(doc_index, 3, 10))

    # Do the same for all the docs, returning a pd DataFrame
    docs_topics_df = best_model.get_dominant_topic_of_document_each_doc_as_df()

    # %%
    # Finding the most representative document for each topic
