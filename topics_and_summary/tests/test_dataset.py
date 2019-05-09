import unittest

from topics_and_summary.datasets.twenty_news_groups import TwentyNewsGroupsDataset, TwentyNewsGroupsDocument
from topics_and_summary.tests.paths import SAVED_OBJECTS_PATH
from topics_and_summary.utils import load_obj_from_disk


class TestTwentyNewsGroupsDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = TwentyNewsGroupsDataset()

    def test_create_dataset(self):
        # Obtain the dataset stored in disk
        expected_dataset = TwentyNewsGroupsDataset.load('dataset', SAVED_OBJECTS_PATH)

        self.assertEqual(expected_dataset, self.dataset)

    def test_get_original_doc_content_from_disk_method(self):
        # Obtain the original content of a specific document
        document = TwentyNewsGroupsDocument(directory_name='alt.atheism', name='51124')
        original_doc_content = self.dataset.get_original_doc_content_from_disk(document)

        expected_original_doc_content = """ 
 
It was no criticism of Islam for a change, it was a criticism of the
arguments used. Namely, whenever people you identify as Muslims are
the victims of the attacks of others, they are used an argument for
the bad situation of Muslims. But whenever deeds by Muslim that victimize
others are named, they do not count as an argument because what these
people did was not done as a true Muslims. No mention is made how Muslims
are the cause of a bad situation of another party."""

        self.assertEqual(expected_original_doc_content, original_doc_content)

    def test_as_documents_content_list_method(self):
        expected_as_documents_content_list = load_obj_from_disk('as_documents_content_list', SAVED_OBJECTS_PATH)
        self.assertEqual(expected_as_documents_content_list, self.dataset.as_documents_content_list())


if __name__ == '__main__':
    unittest.main()
