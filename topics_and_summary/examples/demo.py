import warnings

from topics_and_summary.datasets.twenty_news_groups import TwentyNewsGroupsDataset
from topics_and_summary.models.summarization import TextRank
from topics_and_summary.models.topics import LdaMalletModel, LdaGensimModel
from topics_and_summary.preprocessing.dataset import preprocess_dataset
from topics_and_summary.utils import pretty_print, get_abspath_from_project_source_root
from topics_and_summary.visualizations import plot_word_clouds_of_topics

warnings.filterwarnings('ignore')


def execute():
    """
    Demo of the library functionality.
    """

    # region 1. Load dataset and preprocessing
    pretty_print('1. Load dataset and preprocessing')

    user_input = input('Load previously preprocessed dataset from [d]isk (quick) or '
                       'load dataset and preprocess it in the [m]oment (slow)? (D/m): ')
    if user_input.lower() != 'm':  # D option
        # Load a preprocessed 20newsgroups dataset object (with trigrams)
        preprocessed_dataset = TwentyNewsGroupsDataset.load('trigrams_dataset')
        pretty_print("One of the files of the preprocessed dataset")
        preprocessed_dataset.print_some_files(n=1, print_file_num=False)
    else:  # m option
        # Load the 20newsgroups dataset, applying the dataset specific preprocessing
        # (remove header, remove footer and remove quotes of the documents, as specified
        # in the __init__() default parameters).
        dataset = TwentyNewsGroupsDataset()

        # Prints some files
        pretty_print("One of the files of the dataset after the dataset specific preprocessing")
        dataset.print_some_files(n=1, print_file_num=False)

        # Applies general preprocessing (generating trigrams):
        #   Normalize, lowercase, remove stopwords, remove emails, ...
        #   All this preprocessing and more is applied, as specified in the default parameters
        #   of the preprocess_dataset() function.
        preprocessed_dataset = preprocess_dataset(dataset, ngrams='tri')
        pretty_print("One of the files of the dataset after the preprocessing")
        preprocessed_dataset.print_some_files(n=1, print_file_num=False)
    # endregion

    # region 2. Generate LdaGensimModel or load LdaMalletModel
    pretty_print('2. Generate or load a TopicsModel')

    user_input = input('Load previously generated Lda[M]alletModel (quick op. and better model) or '
                       'generate a Lda[G]ensimModel in the moment (slow op. and worst model)? (M/g): ')
    if user_input.lower() != 'g':  # M option
        model_dir_path = get_abspath_from_project_source_root(
            'saved-elements/topics/best-model/trigrams/lda-mallet')
        # Load a LdaMalletModel stored on disk (the best model found for this dataset)
        # The load() method also loads the dataset used to generate the model, the preprocessing options,
        # and the docs_topics_df DataFrame (contains the dominant topic of each document in the dataset).
        model = LdaMalletModel.load('model17', model_dir_path)
    else:  # g option
        # Generate a LdaGensimModel using the previously preprocessed dataset
        model = LdaGensimModel(preprocessed_dataset, num_topics=17)
    # endregion

    # region 3. Show topics
    pretty_print('3. Show the topics of the chosen model')

    user_input = input('In which format ([t]ext, [i]mages, [b]oth)? (t/i/B):')

    text_format = images_format = False
    if user_input.lower() != 't' and user_input.lower() != 'i':  # B option
        text_format = images_format = True
    elif user_input.lower() == 't':
        text_format = True
    elif user_input.lower() == 'i':
        images_format = True

    if text_format:
        pretty_print('Text format')
        model.print_topics(pretty_format=True)
    if images_format:
        pretty_print('Images')
        print('Images are being saved in the <project-root-path>/demo-images folder')
        # Create a plot with the most important keywords in each topic.
        # Plots are stored in the <project-root-path>/demo-images folder.
        plot_word_clouds_of_topics(
            model.get_topics(num_keywords=15), dpi=150, show_plot=False, save=True,
            dir_save_path=get_abspath_from_project_source_root('../demo-images'))
    # endregion

    # region 4. Get the most representative documents of one topic
    pretty_print('4. Show the k most representative documents of the topic 16')

    k = input('k value (default is 2):')
    try:
        k = int(k)
    except ValueError:
        k = 2

    # Obtain a DataFrame with the k most representative documents of the topic 16
    two_most_repr_docs_topic16_df = model.get_k_most_repr_docs_of_topic_as_df(topic=16, k=k)

    for i in range(k):
        pretty_print('Document {0}'.format(i + 1))
        # The 'Topic prob' column contains the topic-document probability
        print('Probability: {0}'.format(two_most_repr_docs_topic16_df['Topic prob'][i]))

        pretty_print('Original document content')
        # The 'Original doc text' column contains the original text of the documents
        # (the text of the documents before the general preprocessing)
        print(two_most_repr_docs_topic16_df['Original doc text'][i])
    # endregion

    # region 5. Given a text, predict the topics probability
    pretty_print('5. Given a text, predict the topics probability')

    user_input = input('Use a religion [h]ardcoded text or '
                       'write your [o]wn text? (H/o): ')

    if user_input.lower() != 'o':  # H option
        text = """The baptism of Jesus is described in the gospels of Matthew, Mark and Luke. 
John's gospel does not directly describe Jesus' baptism. Most modern theologians view the 
baptism of Jesus by John the Baptist as a historical event to which a high degree of 
certainty can be assigned.[1][2][3][4][5] Along with the crucifixion of Jesus, most biblical 
scholars view it as one of the two historically certain facts about him, and often use it
as the starting point for the study of the historical Jesus.[6] 
The baptism is one of the five major milestones in the gospel narrative of the life of Jesus, 
the others being the Transfiguration, Crucifixion, Resurrection, and Ascension.[7][8] 
Most Christian denominations view the baptism of Jesus as an important event and a basis for 
the Christian rite of baptism (see also Acts 19:1–7). In Eastern Christianity, Jesus' baptism 
is commemorated on 6 January (the Julian calendar date of which corresponds to 19 January on 
the Gregorian calendar), the feast of Epiphany.[9] In the Roman Catholic Church, the Anglican
Communion, the Lutheran Churches and some other Western denominations, it is recalled on a day 
within the following week, the feast of the baptism of the Lord. In Roman Catholicism, 
the baptism of Jesus is one of the Luminous Mysteries sometimes added to the Rosary.
It is a Trinitarian feast in the Eastern Orthodox Churches."""

    else:  # o option
        print('Write your text (when finish, press Enter two times):')
        lines = []
        while True:
            line = input()
            if line:
                lines.append(line)
            else:
                break
        text = '\n'.join(lines)

    pretty_print('Text')
    print(text)

    pretty_print('Text-topics probability')
    # Predict the probability of the text being related with each topic.
    # Instead of storing the returned DataFrame, a table is printed to the standard output
    model.predict_topic_prob_on_text(text)
    # endregion

    # region 6. Given a text, get k most related documents
    pretty_print('6. Given a text, get k most related documents')

    k = input('k value (default is 2):')
    try:
        k = int(k)
    except ValueError:
        k = 2

    pretty_print('Text')
    print(text)

    # Obtain a DataFrame with the k documents more related to the given text
    related_docs_df = model.get_related_docs_as_df(text, num_docs=k)

    for i in range(k):
        pretty_print('Document {0}'.format(i + 1))
        # The 'Doc prob' column contains the document-text probability
        print('Probability: {0}'.format(related_docs_df['Doc prob'][i]))

        pretty_print('Original document content')
        # The 'Original doc text' column contains the original text of the documents
        # (the text of the documents before the general preprocessing)
        print(related_docs_df['Original doc text'][i])
    # endregion

    # region 7. Summarize a given text
    pretty_print('7. Summarize a given text (get k best sentences)')

    k = input('k value (default is 2):')
    try:
        k = int(k)
    except ValueError:
        k = 2

    pretty_print('Text')
    print(text)

    pretty_print('Summary')
    # Create a TextRank model (by default it uses Glove word-embeddings)
    tr = TextRank()
    # Use the created model to obtain the k sentences that better summarize the given text
    summary = tr.get_k_best_sentences_of_text(text, k)

    for i, sent in enumerate(summary):
        if i > 0:
            print()
        print('Sentence {0}: {1}'.format(i + 1, sent))
    # endregion


if __name__ == '__main__':
    """
    This Python module contains an interactive demo of the library functionality.
    The call to the execute() method starts the demo.
    """
    execute()
