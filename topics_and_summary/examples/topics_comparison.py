from typing import List

from topics_and_summary.datasets.twenty_news_groups import TwentyNewsGroupsDataset
from topics_and_summary.models.topics import LdaModelsList, LsaModelsList, LdaMalletModelsList, TopicsModel
from topics_and_summary.preprocessing.dataset.structured import preprocess_dataset
from topics_and_summary.utils import join_paths, pretty_print, get_abspath_from_project_source_root
from topics_and_summary.visualizations import plot_word_clouds_of_topics, tsne_clustering_chart


def generate_and_store_models(path, dataset, plot_first_name):
    # region LDA
    pretty_print(plot_first_name + ' LDA')

    lda_path = join_paths(path, 'lda')
    lda_models_list = LdaModelsList(dataset)
    # Create models, compute coherence values and store a plot with the coherence values
    pretty_print('Creating models')
    lda_models, lda_coherence_values = \
        lda_models_list.create_models_and_compute_coherence_values(MIN_TOPICS, MAX_TOPICS,
                                                                   title=plot_first_name + ' LDA models',
                                                                   save_plot=True,
                                                                   save_plot_path=join_paths(lda_path,
                                                                                             'coherence_values.png'))
    # Store the models and a txt file with the coherence value of each model
    pretty_print('Storing models')
    lda_models_list.save(base_name='model', path=lda_path)
    store_plots(lda_models, lda_coherence_values)
    # endregion

    # region LSA
    pretty_print(plot_first_name + ' LSA')

    lsa_path = join_paths(path, 'lsa')
    lsa_models_list = LsaModelsList(dataset)
    # Create models, compute coherence values and store a plot with the coherence values
    pretty_print('Creating models')
    lsa_models, lsa_coherence_values = \
        lsa_models_list.create_models_and_compute_coherence_values(MIN_TOPICS, MAX_TOPICS,
                                                                   title=plot_first_name + ' LSA models',
                                                                   save_plot=True,
                                                                   save_plot_path=join_paths(lsa_path,
                                                                                             'coherence_values.png'))
    # Store the models and a txt file with the coherence value of each model
    pretty_print('Storing models')
    lsa_models_list.save(base_name='model', path=lsa_path)
    # endregion

    # region LDA Mallet
    pretty_print(plot_first_name + ' LDA Mallet')

    lda_mallet_path = join_paths(path, 'lda-mallet')
    lda_mallet_models_list = LdaMalletModelsList(dataset)
    # Create models, compute coherence values and store a plot with the coherence values
    pretty_print('Creating models')
    lda_mallet_models, lda_mallet_coherence_values = \
        lda_mallet_models_list.create_models_and_compute_coherence_values(MIN_TOPICS, MAX_TOPICS,
                                                                          title=plot_first_name + ' LDA Mallet models',
                                                                          save_plot=True,
                                                                          save_plot_path=join_paths(lda_mallet_path,
                                                                                                    'coherence_values.png'),
                                                                          models_base_name='model',
                                                                          model_path=lda_mallet_path)
    # Store the models and a txt file with the coherence value of each model
    pretty_print('Storing models')
    lda_mallet_models_list.save()
    # tSNE is too slow to calculate, because predictions in LdaMallet are too slow
    store_plots(lda_mallet_models, lda_mallet_coherence_values, tsne=False)
    # endregion


def store_plots(models: List[TopicsModel], coherence_values: List[float], tsne=True):
    """
    Given a list of models and a list of coherence values, stores the plots of the wordclouds \
    and the tsne html interactive plot in the dir_path of the model with max coherence value.

    :param tsne: If true, calculates tsne and stores plot.
    """
    pretty_print('Storing plots')

    # Get the best model using the coherence value
    index_max_coherence_value = coherence_values.index(max(coherence_values))
    best_model = models[index_max_coherence_value]

    # Store the wordclouds plots of only the bests models of the list
    plot_word_clouds_of_topics(best_model.get_topics(), save=True, dir_save_path=best_model.dir_path,
                               show_plot=False, dpi=100)

    if tsne:
        # Store the tSNE plot of only the best model of the list
        tsne_clustering_chart(best_model, save_path=best_model.dir_path, plot_name='tsne.html', show_plot=False)


if __name__ == '__main__':
    """
    This Python module generates a combination of different models (the ones describe below), 
    and generates some plots based on the coherence values and the topics generated,
    making it easier to compare models and select the best one.

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
    BASE_PATH = get_abspath_from_project_source_root('saved-elements/topics/comparison')

    # %%
    # Unigrams
    pretty_print('Unigrams')
    unigrams_dataset = preprocess_dataset(dataset, ngrams='uni')
    unigrams_path = join_paths(BASE_PATH, 'unigrams')

    generate_and_store_models(unigrams_path, unigrams_dataset, 'Unigrams')

    # Bigrams
    pretty_print('Bigrams')
    bigrams_dataset = preprocess_dataset(dataset, ngrams='bi')
    bigrams_path = join_paths(BASE_PATH, 'bigrams')

    generate_and_store_models(bigrams_path, bigrams_dataset, 'Bigrams')

    # Trigrams
    pretty_print('Trigrams')
    trigrams_dataset = preprocess_dataset(dataset, ngrams='tri')
    trigrams_path = join_paths(BASE_PATH, 'trigrams')

    generate_and_store_models(trigrams_path, trigrams_dataset, 'Trigrams')
