from models.topics import LdaMalletModel
from utils import save_obj_to_disk, load_obj_from_disk, get_abspath_from_project_root

if __name__ == '__main__':
    # Uncomment one of this 2 lines:
    trigrams_dataset = load_obj_from_disk('trigrams_dataset')  # Dataset already preprocessed and stored on disk
    # trigrams_dataset = preprocess_dataset(TwentyNewsGroupsDataset(), ngrams='tri')  # Dataset is preprocessed now

    documents = trigrams_dataset.as_documents_content_list()

    path = get_abspath_from_project_root('saved-elements/topics/best-model/trigrams/lda-mallet/')
    model = LdaMalletModel.load('model17', documents, path)

    docs_topics_df = model.get_dominant_topic_of_each_doc_as_df()
    save_obj_to_disk(docs_topics_df, 'mallet_17topics_docs_topics_df')