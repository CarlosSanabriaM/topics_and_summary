from topics_and_summary.datasets.twenty_news_groups import TwentyNewsGroupsDataset
from topics_and_summary.models.topics import LdaGensimModel
from topics_and_summary.preprocessing.dataset import preprocess_dataset
from topics_and_summary.utils import pretty_print, RANDOM_STATE
from topics_and_summary.visualizations import plot_word_clouds_of_topics, tsne_clustering_chart

if __name__ == '__main__':
    """
    This Python module shows some of the functionalities of the library.
    """
    # %%

    # Load dataset and apply preprocessing
    dataset = TwentyNewsGroupsDataset()
    dataset = preprocess_dataset(dataset, ngrams='tri')

    # Create the Lda model
    pretty_print('Creating the Lda model')
    model = LdaGensimModel(dataset, num_topics=20, random_state=RANDOM_STATE)

    # Visualize with tsne
    tsne_clustering_chart(model)

    # %%

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
    plot_word_clouds_of_topics(model.get_topics())

    docs_topics_df = model.get_dominant_topic_of_each_doc_as_df()
    three_most_repr_doc_per_topic_df = model.get_k_most_repr_docs_per_topic_as_df(k=3)
    topic_distribution_df = model.get_topic_distribution_as_df()

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
    the disk space very well. But sometimes, it has many issues."""

    text3 = """Car gas speed wheels Windows MSDOS issue space jesus god bible people mac U.S.A. guns lebanese"""

    text4 = """Penicillin (PCN or pen) is a group of antibiotics which include penicillin G (intravenous use), 
    penicillin V (use by mouth), procaine penicillin, and benzathine penicillin (intramuscular use). 
    Penicillin antibiotics were among the first medications to be effective against many bacterial infections caused by 
    staphylococci and streptococci. They are still widely used today, though many types of bacteria have developed 
    resistance following extensive use."
    About 10% of people report that they are allergic to penicillin; however, up to 90% of this group may not
    actually be allergic.[2] Serious allergies only occur in about 0.03%.[2] All penicillins are β-lactam
    antibiotics."
    Penicillin was discovered in 1928 by Scottish scientist Alexander Fleming.[3] People began using
    it to treat infections in 1942.[4] There are several enhanced penicillin families which are effective
    against additional bacteria; these include the antistaphylococcal penicillins, aminopenicillins and the
    antipseudomonal penicillins. They are derived from Penicillium fungi.[5]"""

    text5 = """"The violence, corruption, and abuse in Central American countries tend to be the biggest factors driving
    migration to the United States—a phenomenon the Trump administration has dedicated itself to curbing.
    Since the gun sales fuel the violence and corruption, the United States has effectively undermined its own
    objectives by allowing the weapons deals, according to experts."""

    related_docs_text1_df = model.get_related_docs_as_df(text1, num_docs=3)
    related_docs_text2_df = model.get_related_docs_as_df(text2, num_docs=3)
    related_docs_text3_df = model.get_related_docs_as_df(text3, num_docs=3)
