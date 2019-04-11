import unittest

from datasets.twenty_news_groups import TwentyNewsGroupsDataset, TwentyNewsGroupsDocument
from utils import load_obj_from_disk, get_abspath_from_project_root, join_paths


class TestTwentyNewsGroupsDataset(unittest.TestCase):
    TESTS_BASE_PATH = get_abspath_from_project_root('tests')

    @classmethod
    def setUpClass(cls):
        cls.dataset = TwentyNewsGroupsDataset()

    def test_create_dataset(self):
        # Obtain the dataset stored in disk
        dataset_from_disk = load_obj_from_disk('dataset',
                                               join_paths(self.TESTS_BASE_PATH, 'saved-elements/objects'))

        self.assertEqual(self.dataset, dataset_from_disk)

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

        self.assertEqual(original_doc_content, expected_original_doc_content)

    def test_as_documents_obj_list_method(self):
        expected_as_documents_obj_list = load_obj_from_disk('as_documents_obj_list',
                                                            join_paths(self.TESTS_BASE_PATH, 'saved-elements/objects'))
        self.assertEqual(self.dataset.as_documents_obj_list(), expected_as_documents_obj_list)

    def test_as_documents_content_list_method(self):
        expected_as_documents_content_list = \
            load_obj_from_disk('as_documents_content_list', join_paths(self.TESTS_BASE_PATH, 'saved-elements/objects'))
        self.assertEqual(self.dataset.as_documents_content_list(), expected_as_documents_content_list)


if __name__ == '__main__':
    unittest.main()
