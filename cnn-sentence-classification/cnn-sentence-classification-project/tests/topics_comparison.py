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
    three_most_repr_doc_per_topic_df = model.get_k_most_representative_docs_per_topic_as_df(k=3)

    text1 = """The baptism of Jesus is described in the gospels of Matthew, Mark and Luke. John's gospel does not
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

    text2 = """Windows 10 is a very good operating system. Many files can be opened at the same time, and it manages
    the disk space very well."""

    text3 = """Car gas speed wheels Windows MSDOS issue space jesus god bible people mac U.S.A. guns lebanese"""

    related_docs_text1_df = model.get_related_documents_as_df(text1, k_docs_per_topic=3)
    related_docs_text2_df = model.get_related_documents_as_df(text2, k_docs_per_topic=3)
    related_docs_text3_df = model.get_related_documents_as_df(text3, k_docs_per_topic=3)
