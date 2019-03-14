from datasets.twenty_news_groups import TwentyNewsGroupsDataset
from models.topics import LdaGensimModel
from preprocessing.dataset import preprocess_dataset
from utils import pretty_print, RANDOM_STATE
from visualizations import plot_word_clouds_k_keywords_each_topic

if __name__ == '__main__':
    # Load dataset and apply preprocessing
    dataset = TwentyNewsGroupsDataset()
    dataset = preprocess_dataset(dataset)

    # Create the Lda model
    pretty_print('Creating the Lda model')
    documents = dataset.as_documents_list()
    model = LdaGensimModel(documents, num_topics=20, random_state=RANDOM_STATE)

    # Print topics and coherence score
    pretty_print('\nTopics')
    NUM_WORDS_EACH_TOPIC_TO_BE_PRINTED = 15
    model.print_topics(NUM_WORDS_EACH_TOPIC_TO_BE_PRINTED)
    coherence_score = model.compute_coherence_value()
    pretty_print('Coherence Score')
    print(coherence_score)

    # Save model to disk.
    model.save('lda_test')

    # Get insights about the topics
    plot_word_clouds_k_keywords_each_topic(model)

    docs_topics_df = model.get_dominant_topic_of_each_doc_as_df()
    most_repr_doc_per_topic_df = model.get_most_representative_doc_per_topic_as_df()
