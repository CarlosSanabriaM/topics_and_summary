from topics_and_summary.models.summarization import TextRank
from topics_and_summary.models.topics import LdaMalletModel, LdaGensimModel
from topics_and_summary.datasets.twenty_news_groups import TwentyNewsGroupsDataset
from topics_and_summary.preprocessing.dataset import preprocess_dataset
from topics_and_summary.utils import pretty_print, load_obj_from_disk, load_func_from_disk, \
    get_abspath_from_project_source_root
from topics_and_summary.visualizations import plot_word_clouds_of_topics, tsne_clustering_chart

if __name__ == '__main__':
    """
    This file contains a demo of the topics functionality.
    """

    # region 1. Load dataset and preprocessing
    pretty_print('1. Load dataset and preprocessing')

    user_input = input('Load previously preprocessed dataset from disk (quick) or '
                       'load dataset and preprocess it in the moment (slow)? (Y/n): ')
    if user_input.lower() != 'n':  # Y option
        dataset = TwentyNewsGroupsDataset.load('trigrams_dataset')
        trigrams_func = load_func_from_disk('trigrams_func')
        pretty_print("One of the files of the preprocessed dataset")
        dataset.print_some_files(n=1, print_file_num=False)
    else:  # n option
        # Load 20 newsgroups dataset, applying the specific preprocessing (specified in the default parameters)
        dataset = TwentyNewsGroupsDataset()

        # Prints some files
        pretty_print("One of the files of the dataset after the dataset specific preprocessing")
        dataset.print_some_files(n=1, print_file_num=False)

        # Applies preprocessing, generating trigrams
        # All the parameters except ngrams have their default value
        dataset, trigrams_func = preprocess_dataset(dataset, ngrams='tri')
        pretty_print("One of the files of the dataset after the preprocessing")
        dataset.print_some_files(n=1, print_file_num=False)
    # endregion

    # region 2. Generate LdaGensimModel or user_input LdaMalletModel
    pretty_print('2. Generate or load a TopicsModel')

    user_input = input('Load previously generated LdaMalletModel (quick and better model) or '
                       'generate LdaGensimModel in the moment (slow and worst model)? (M/g): ')
    if user_input.lower() != 'g':  # M option
        model_dir_path = get_abspath_from_project_source_root('saved-elements/topics/best-model/trigrams/lda-mallet')
        docs_topics_df = load_obj_from_disk('mallet_17topics_docs_topics_df')
        model = LdaMalletModel.load('model17', dataset, model_dir_path, docs_topics_df=docs_topics_df)
    else:  # g option
        model = LdaGensimModel(dataset, num_topics=17)
    # endregion

    # region 2. Show topics
    pretty_print('2. Show the topics of the chosen model')

    user_input = input('In which format (text, images, both)? (t/i/B):')
    if user_input.lower() != 't' and user_input.lower() != 'i':  # B option
        pretty_print('Text format')
        model.print_topics()
        pretty_print('Images')
        plot_word_clouds_of_topics(model.get_topics(num_keywords=20))
    elif user_input.lower() == 't':
        pretty_print('Text format')
        model.print_topics()
    elif user_input.lower() == 'i':
        pretty_print('Images')
        plot_word_clouds_of_topics(model.get_topics(num_keywords=20))
    # endregion

    # region 3. Most repr docs of one topic
    pretty_print('3. Show the k most representative documents of the topic 16')

    k = input('k value (default is 2):')
    try:
        k = int(k)
    except ValueError:
        k = 2

    two_most_repr_docs_topic16_df = model.get_k_most_repr_docs_of_topic_as_df(16, k=k)
    for i in range(k):
        pretty_print('Document {0}'.format(i + 1))
        print('Probability: {0}'.format(two_most_repr_docs_topic16_df['Topic prob'][i]))
        pretty_print('Original document content')
        print(two_most_repr_docs_topic16_df['Original doc text'][i])

    # endregion

    # region 4. Given a text, predict the topics probability
    pretty_print('4. Given a text, predict the topics probability')

    input('Press any key')

    text = """The baptism of Jesus is described in the gospels of Matthew, Mark and Luke. John's gospel does not
directly describe Jesus' baptism. Most modern theologians view the baptism of Jesus by John the Baptist as a
historical event to which a high degree of certainty can be assigned.[1][2][3][4][5] Along with the crucifixion
of Jesus, most biblical scholars view it as one of the two historically certain facts about him, and often use it
as the starting point for the study of the historical Jesus.[6]
The baptism is one of the five major milestones in the gospel narrative of the life of Jesus, the others being
the Transfiguration, Crucifixion, Resurrection, and Ascension.[7][8] Most Christian denominations view the baptism
of Jesus as an important event and a basis for the Christian rite of baptism (see also Acts 19:1–7).
In Eastern Christianity, Jesus' baptism is commemorated on 6 January (the Julian calendar date of which corresponds
to 19 January on the Gregorian calendar), the feast of Epiphany.[9] In the Roman Catholic Church, the Anglican
Communion, the Lutheran Churches and some other Western denominations, it is recalled on a day within the following
week, the feast of the baptism of the Lord. In Roman Catholicism, the baptism of Jesus is one of the Luminous
Mysteries sometimes added to the Rosary. It is a Trinitarian feast in the Eastern Orthodox Churches."""

    pretty_print('Text')
    print(text)

    pretty_print('Text-topics probability')
    model.predict_topic_prob_on_text(text, ngrams='tri', ngrams_model_func=trigrams_func)
    # endregion

    # region 5. Given a text, get k most related documents
    pretty_print('5. Given a text, get k most related documents')

    k = input('k value (default is 2):')
    try:
        k = int(k)
    except ValueError:
        k = 2

    pretty_print('Text')
    print(text)

    related_docs_df = model.get_related_docs_as_df(text, num_docs=k, ngrams='tri', ngrams_model_func=trigrams_func)
    for i in range(k):
        pretty_print('Document {0}'.format(i + 1))
        print('Probability: {0}'.format(related_docs_df['Doc prob'][i]))
        pretty_print('Original document content')
        print(related_docs_df['Original doc text'][i])
    # endregion

    # region 6. Summarize a given text
    pretty_print('6. Summarize a given text (get k best sentences)')

    k = input('k value (default is 2):')
    try:
        k = int(k)
    except ValueError:
        k = 2

    pretty_print('Text')
    print(text)

    pretty_print('Summary')
    tr = TextRank()
    summary = tr.get_k_best_sentences_of_text(text, k)
    for i, sent in enumerate(summary):
        print('Sentence {0}: {1}'.format(i + 1, sent))

    # endregion