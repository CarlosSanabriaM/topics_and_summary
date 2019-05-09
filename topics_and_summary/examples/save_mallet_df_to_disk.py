from topics_and_summary.datasets.twenty_news_groups import TwentyNewsGroupsDataset
from topics_and_summary.models.topics import LdaMalletModel
from topics_and_summary.utils import get_abspath_from_project_source_root

if __name__ == '__main__':
    path = get_abspath_from_project_source_root('saved-elements/topics/best-model/trigrams/lda-mallet/')
    model = LdaMalletModel.load('model17', TwentyNewsGroupsDataset, path)

    model.get_dominant_topic_of_each_doc_as_df()
    model.save()
