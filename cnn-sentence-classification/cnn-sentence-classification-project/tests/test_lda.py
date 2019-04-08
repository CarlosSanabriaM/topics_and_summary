from datasets.twenty_news_groups import TwentyNewsGroupsDataset
from models.topics import LdaGensimModel
from preprocessing.dataset import preprocess_dataset
from preprocessing.text import preprocess_text
from utils import pretty_print, RANDOM_STATE

if __name__ == '__main__':
    # %%
    # Load dataset and apply preprocessing
    dataset = TwentyNewsGroupsDataset()
    dataset = preprocess_dataset(dataset)

    # %%
    # Create the Lda model
    pretty_print('Creating the Lda model')
    documents = dataset.as_documents_content_list()
    model = LdaGensimModel(documents, num_topics=20, random_state=RANDOM_STATE)

    # %%
    # Print topics and coherence score
    pretty_print('\nTopics')
    NUM_WORDS_EACH_TOPIC_TO_BE_PRINTED = 15
    model.print_topics(NUM_WORDS_EACH_TOPIC_TO_BE_PRINTED)
    coherence_score = model.compute_coherence_value()
    pretty_print('Coherence Score')
    print(coherence_score)

    # %%
    # Save model to disk.
    model.save('lda_test')

    # %%
    # Get insights about the topics
    docs_topics_df = model.get_dominant_topic_of_each_doc_as_df()
    most_repr_doc_per_topic_df = model.get_k_most_representative_docs_per_topic_as_df()

    # %%
    # Query the model using new, unseen documents
    pretty_print("Query the model using new, unseen documents")
    test_docs = [
        "Windows MSDOS was installed in a very bad hardware. It had many issues.",

        "Penicillin (PCN or pen) is a group of antibiotics which include penicillin G (intravenous use), penicillin V "
        "(use by mouth), procaine penicillin, and benzathine penicillin (intramuscular use). Penicillin antibiotics "
        "were among the first medications to be effective against many bacterial infections caused by staphylococci "
        "and streptococci. They are still widely used today, though many types of bacteria have developed resistance "
        "following extensive use."
        "About 10% of people report that they are allergic to penicillin; however, up to 90% of this group may not "
        "actually be allergic.[2] Serious allergies only occur in about 0.03%.[2] All penicillins are β-lactam "
        "antibiotics."
        "Penicillin was discovered in 1928 by Scottish scientist Alexander Fleming.[3] People began using "
        "it to treat infections in 1942.[4] There are several enhanced penicillin families which are effective "
        "against additional bacteria; these include the antistaphylococcal penicillins, aminopenicillins and the "
        "antipseudomonal penicillins. They are derived from Penicillium fungi.[5]",

        "The violence, corruption, and abuse in Central American countries tend to be the biggest factors driving "
        "migration to the United States—a phenomenon the Trump administration has dedicated itself to curbing. "
        "Since the gun sales fuel the violence and corruption, the United States has effectively undermined its own "
        "objectives by allowing the weapons deals, according to experts."
    ]

    pretty_print("Texts before preprocessing")
    for doc in test_docs:
        print('\n' + doc)

    preprocessed_test_docs = [preprocess_text(doc) for doc in test_docs]

    pretty_print("Texts after preprocessing")
    for doc in preprocessed_test_docs:
        print('\n' + doc)

    # get topic probability distribution for each document
    print("Topic prob vector doc 0:", model.predict_topic_prob_on_text(preprocessed_test_docs[0], preprocess=False))
    print("Topic prob vector doc 1:", model.predict_topic_prob_on_text(preprocessed_test_docs[1], preprocess=False))
    print("Topic prob vector doc 2:", model.predict_topic_prob_on_text(preprocessed_test_docs[2], preprocess=False))

    # %%
    # Update the model by incrementally training on the new corpus (Online training)
    # model.update(test_corpus)
    # topic_prob_vector_test_doc_1_updated = model[test_corpus[1]]

    # %%
    # Load model from disk
    model_from_disk = LdaGensimModel.load(
        'lda_test_2019-03-13 08:49:25.022155 topics_20 coherence_0.4666285571172143', documents)

    print(model_from_disk.num_topics)
    print(model_from_disk.compute_coherence_value())

    # %%
    # Using the per_word_topics option while creating gensim lda model. This option is only available in lda model.
    model_using_per_word_topics = LdaGensimModel(documents, num_topics=20, random_state=RANDOM_STATE,
                                                 per_word_topics=True)

    # Doesn't seem to be very useful
    model_using_per_word_topics.model.get_document_topics(model.dictionary.doc2bow(preprocessed_test_docs[0].split()),
                                                          per_word_topics=True)
    """
    Out[11]:
    ([(13, 0.90500003)],
     [(664, [13]),
      (1068, [13]),
      (1076, [13]),
      (1853, [13]),
      (3604, [13]),
      (10060, [13]),
      (15219, [13]),
      (17171, [13]),
      (17271, [13])],
     [(664, [(13, 1.0)]),
      (1068, [(13, 0.99999994)]),
      (1076, [(13, 1.0)]),
      (1853, [(13, 0.99999994)]),
      (3604, [(13, 0.99999994)]),
      (10060, [(13, 1.0)]),
      (15219, [(13, 1.0)]),
      (17171, [(13, 1.0)]),
      (17271, [(13, 1.0)])])
    """
