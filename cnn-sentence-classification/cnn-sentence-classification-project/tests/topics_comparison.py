from typing import List

from models.topics import LdaModelsList, LsaModelsList, LdaMalletModelsList, TopicsModel
from preprocessing.dataset import preprocess_dataset
from utils import get_abspath, join_paths, pretty_print, load_obj_from_disk
from visualizations import plot_word_clouds_k_keywords_each_topic, tsne_clustering_chart


def generate_and_store_models(path, documents, plot_first_name):
    # region LDA
    pretty_print(plot_first_name + ' LDA')

    lda_path = join_paths(path, 'lda')
    lda_models_list = LdaModelsList(documents)
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
    lsa_models_list = LsaModelsList(documents)
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
    # store_plots(lsa_models, lsa_coherence_values)  # TODO: Raises Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
    # endregion

    # region LDA Mallet
    pretty_print(plot_first_name + ' LDA Mallet')

    lda_mallet_path = join_paths(path, 'lda-mallet')
    lda_mallet_models_list = LdaMalletModelsList(documents)
    # Create models, compute coherence values and store a plot with the coherence values
    pretty_print('Creating models')
    lda_mallet_models, lda_mallet_coherence_values = \
        lda_mallet_models_list.create_models_and_compute_coherence_values(MIN_TOPICS, MAX_TOPICS,
                                                                          title=plot_first_name + ' LDA Mallet models',
                                                                          save_plot=True,
                                                                          save_plot_path=join_paths(lda_mallet_path,
                                                                                                    'coherence_values.png'),
                                                                          models_base_name='model')
    # Store the models and a txt file with the coherence value of each model
    pretty_print('Storing models')
    lda_mallet_models_list.save()
    store_plots(lda_mallet_models, lda_mallet_coherence_values)
    # endregion


def store_plots(models: List[TopicsModel], coherence_values: List[float]):
    """
    Given a list of models and a list of coherence values, stores the plots of the wordclouds
    and the tsne html interactive plot in the dir_path of the model with max coherence value.
    """
    pretty_print('Storing plots')

    # Get the best model using the coherence value
    index_max_coherence_value = coherence_values.index(max(coherence_values))
    best_model = models[index_max_coherence_value]

    # Store the wordclouds plots of only the bests models of the list
    plot_word_clouds_k_keywords_each_topic(best_model, save=True, dir_save_path=best_model.dir_path,
                                           show_plot=False, dpi=100)

    # Store the tSNE plot of only the best model of the list
    tsne_clustering_chart(best_model, save_path=best_model.dir_path, plot_name='tsne.html', show_plot=False)


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
    # dataset = TwentyNewsGroupsDataset() # TODO: Uncomment

    # Topics info for the models
    # TODO: Uncomment
    # MIN_TOPICS = 10
    # MAX_TOPICS = 20
    MIN_TOPICS = 2  # TODO: Remove
    MAX_TOPICS = 3  # TODO: Remove
    BASE_PATH = '../saved-models/topics/comparison'

    # %%
    # Unigrams
    pretty_print('Unigrams')
    # unigrams_dataset = preprocess_dataset(dataset, ngrams='uni')  # TODO: Uncomment
    unigrams_dataset = load_obj_from_disk('unigrams_dataset')  # TODO: Remove
    unigrams_documents = unigrams_dataset.as_documents_list()
    unigrams_path = get_abspath(__file__, join_paths(BASE_PATH, 'unigrams'))

    generate_and_store_models(unigrams_path, unigrams_documents, 'Unigrams')

    # Bigrams
    pretty_print('Bigrams')
    bigrams_dataset = preprocess_dataset(dataset, ngrams='bi')
    bigrams_documents = bigrams_dataset.as_documents_list()
    bigrams_path = get_abspath(__file__, join_paths(BASE_PATH, 'bigrams'))

    generate_and_store_models(bigrams_path, bigrams_documents, 'Bigrams')

    # Trigrams
    pretty_print('Trigrams')
    trigrams_dataset = preprocess_dataset(dataset, ngrams='tri')
    trigrams_documents = trigrams_dataset.as_documents_list()
    trigrams_path = get_abspath(__file__, join_paths(BASE_PATH, 'trigrams'))

    generate_and_store_models(trigrams_path, trigrams_documents, 'Trigrams')
