from topics_and_summary.datasets.twenty_news_groups import TwentyNewsGroupsDataset
from topics_and_summary.models.topics import LdaMalletModel
from topics_and_summary.utils import get_abspath_from_project_source_root

if __name__ == '__main__':
    """
    This Python module loads from disk the best model (in this case, a lda-mallet model with 17 topics)
    and generates the docs_topics_df, calling the get_dominant_topic_of_each_doc_as_df() method.
    The save() method stores in disk the generate docs_topics_df inside the model folder.
    """
    path = get_abspath_from_project_source_root('saved-elements/topics/best-model/trigrams/lda-mallet/')
    model = LdaMalletModel.load('model17', TwentyNewsGroupsDataset, path)

    model.get_dominant_topic_of_each_doc_as_df()
    model.save()
