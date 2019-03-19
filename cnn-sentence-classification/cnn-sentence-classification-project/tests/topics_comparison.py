from datasets.twenty_news_groups import TwentyNewsGroupsDataset
from models.topics import LdaModelsList
from preprocessing.dataset import preprocess_dataset
from utils import get_abspath
from visualizations import plot_word_clouds_k_keywords_each_topic, tsne_clustering_chart

if __name__ == '__main__':
    """
   Models to be created and compared:
   1. Unigrams
       1.1 Models with LDA between 10 and 20 topics
       1.2 Models with LSA between 10 and 20 topics
       1.3 Models with LDA Mallet between 10 and 20 topics
   2. Bigrams
       2.1 Models with LDA between 10 and 20 topics
       2.2 Models with LSA between 10 and 20 topics
       2.3 Models with LDA Mallet between 10 and 20 topics
   3. Trigrams
       3.1 Models with LDA between 10 and 20 topics
       3.2 Models with LSA between 10 and 20 topics
       3.3 Models with LDA Mallet between 10 and 20 topics
   """

    # %%
    # Load dataset
    dataset = TwentyNewsGroupsDataset()

    # Topics info for the models
    MIN_TOPICS = 10
    MAX_TOPICS = 20
    BASE_PATH = '../saved-models/topics/comparison/'

    # %%
    # Unigrams
    unigrams_path = get_abspath(__file__, BASE_PATH + 'unigrams/')
    unigrams_dataset = preprocess_dataset(dataset, ngrams='uni')
    unigrams_documents = unigrams_dataset.as_documents_list()

    # region LDA
    unigrams_lda_path = unigrams_path + 'lda/'
    unigrams_lda_models_list = LdaModelsList(unigrams_documents)
    # Create models, compute coherence values and store a plot with the coherence values
    unigrams_lda_models, unigrams_lda_coherence_values = \
        unigrams_lda_models_list.create_models_and_compute_coherence_values(MIN_TOPICS, MAX_TOPICS,
                                                                            title='Unigrams LDA models',
                                                                            save_plot=True,
                                                                            save_plot_path=unigrams_lda_path + 'coherence_values.png')
    # Store the models and a txt file with the coherence value of each model
    unigrams_lda_models_list.save('model')

    # Store the word clouds plot
    for model in unigrams_lda_models:
        plot_word_clouds_k_keywords_each_topic(model, save=True, dir_save_path=model.dir_path)

    # Store the tSNE plot
    for model in unigrams_lda_models:
        tsne_clustering_chart(model, save_path=model.dir_path, plot_name='tsne')

    # endregion
