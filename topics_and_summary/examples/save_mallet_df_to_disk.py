from topics_and_summary.datasets.twenty_news_groups import TwentyNewsGroupsDataset
from topics_and_summary.models.topics import LdaMalletModel
from topics_and_summary.utils import save_obj_to_disk, load_obj_from_disk, get_abspath_from_project_root

if __name__ == '__main__':
    # Uncomment one of this 2 lines:
    trigrams_dataset = TwentyNewsGroupsDataset.load('trigrams_dataset')  # Dataset already preprocessed
    # trigrams_dataset = preprocess_dataset(TwentyNewsGroupsDataset(), ngrams='tri')  # Dataset is preprocessed now

    documents = trigrams_dataset.as_documents_content_list()

    path = get_abspath_from_project_root('saved-elements/topics/best-model/trigrams/lda-mallet/')
    model = LdaMalletModel.load('model17', documents, path)

    docs_topics_df = model.get_dominant_topic_of_each_doc_as_df()
    save_obj_to_disk(docs_topics_df, 'mallet_17topics_docs_topics_df')
