from datasets.twenty_news_groups import TwentyNewsGroupsDataset
from models.topics import LdaModelsList

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
    documents = dataset.as_documents_list()

    # Topics info for the models
    MIN_TOPICS = 10
    MAX_TOPICS = 20

    # %%
    # Unigrams
    unigrams_lda_models_list = LdaModelsList(documents)
    unigrams_lda_models_list.create_models_and_compute_coherence_values(MIN_TOPICS, MAX_TOPICS, )
