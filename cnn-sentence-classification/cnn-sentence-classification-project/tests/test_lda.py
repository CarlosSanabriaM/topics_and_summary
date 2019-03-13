from datasets.twenty_news_groups import TwentyNewsGroupsDataset
from models.topics import LdaGensimModel
from preprocessing.dataset import preprocess_dataset
from preprocessing.text import preprocess_text
from utils import pretty_print, get_abspath, RANDOM_STATE

if __name__ == '__main__':
    # %%
    # Load dataset and apply preprocessing
    dataset = TwentyNewsGroupsDataset()
    dataset = preprocess_dataset(dataset)

    # %%
    # Create the Lda model
    pretty_print('Creating the Lda model')
    documents = dataset.as_documents_list()
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
    print("Topic prob vector doc 0:", model.predict_topic_prob_on_text(preprocessed_test_docs[0]))
    print("Topic prob vector doc 1:", model.predict_topic_prob_on_text(preprocessed_test_docs[1]))
    print("Topic prob vector doc 2:", model.predict_topic_prob_on_text(preprocessed_test_docs[2]))

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

# TODO:
"""
per_word_topics (bool) – If True, the model also computes a list of topics, 
sorted in descending order of most likely topics for each word, along with 
their phi values multiplied by the feature length (i.e. word count).
"""
