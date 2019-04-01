from models.topics import LdaMalletModel
from utils import save_obj_to_disk, load_obj_from_disk

if __name__ == '__main__':
    trigrams_dataset = load_obj_from_disk('trigrams_dataset')  # Change if new preprocessing is applied
    documents = trigrams_dataset.as_documents_list()

    model = LdaMalletModel.load('model17', documents, 'saved-models/topics/comparison/trigrams/lda-mallet/')

    docs_topics_df = model.get_dominant_topic_of_each_doc_as_df()
    save_obj_to_disk(docs_topics_df, 'mallet_17topics_docs_topics_df')
