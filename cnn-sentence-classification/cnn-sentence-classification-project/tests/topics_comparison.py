from datasets.twenty_news_groups import TwentyNewsGroupsDataset
from models.topics import LdaModelsList, LsaModelsList, LdaMalletModelsList
from preprocessing.dataset import preprocess_dataset
from utils import get_abspath
from visualizations import plot_word_clouds_k_keywords_each_topic, tsne_clustering_chart


def generate_and_store_models(path, documents, plot_first_name):
    # region LDA
    lda_path = path + 'lda/'
    lda_models_list = LdaModelsList(documents)
    # Create models, compute coherence values and store a plot with the coherence values
    lda_models, _ = lda_models_list.create_models_and_compute_coherence_values(MIN_TOPICS, MAX_TOPICS,
                                                                               title=plot_first_name + ' LDA models',
                                                                               save_plot=True,
                                                                               save_plot_path=lda_path + 'coherence_values.png')
    # Store the models and a txt file with the coherence value of each model
    lda_models_list.save('model')
    store_plots(lda_models)
    # endregion

    # region LSA
    lsa_path = path + 'lsa/'
    lsa_models_list = LsaModelsList(documents)
    # Create models, compute coherence values and store a plot with the coherence values
    lsa_models, _ = lsa_models_list.create_models_and_compute_coherence_values(MIN_TOPICS, MAX_TOPICS,
                                                                               title=plot_first_name + ' LSA models',
                                                                               save_plot=True,
                                                                               save_plot_path=lsa_path + 'coherence_values.png')
    # Store the models and a txt file with the coherence value of each model
    lsa_models_list.save('model')
    store_plots(lsa_models)
    # endregion

    # region LDA Mallet
    lda_mallet_path = path + 'lda-mallet/'
    lda_mallet_models_list = LdaMalletModelsList(documents)
    # Create models, compute coherence values and store a plot with the coherence values
    lda_mallet_models, _ = \
        lda_mallet_models_list.create_models_and_compute_coherence_values(MIN_TOPICS, MAX_TOPICS,
                                                                          title=plot_first_name + ' LDA Mallet models',
                                                                          save_plot=True,
                                                                          save_plot_path=lda_mallet_path + 'coherence_values.png',
                                                                          models_base_name='model')
    # Store the models and a txt file with the coherence value of each model
    lda_mallet_models_list.save()
    store_plots(lda_mallet_models)
    # endregion


def store_plots(models):
    # Store the word clouds plot
    for model in models:
        plot_word_clouds_k_keywords_each_topic(model, save=True, dir_save_path=model.dir_path)
    # Store the tSNE plot
    for model in models:
        tsne_clustering_chart(model, save_path=model.dir_path, plot_name='tsne')


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

    generate_and_store_models(unigrams_path, unigrams_documents, 'Unigrams')
