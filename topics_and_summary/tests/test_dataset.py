import unittest
from copy import deepcopy
from shutil import rmtree

from topics_and_summary.datasets.twenty_news_groups import TwentyNewsGroupsDataset, TwentyNewsGroupsDocument
from topics_and_summary.preprocessing.dataset_preprocessing_options import DatasetPreprocessingOptions
from topics_and_summary.tests.paths import SAVED_OBJECTS_PATH, SAVED_FUNCS_PATH
from topics_and_summary.utils import load_obj_from_disk, load_func_from_disk, join_paths


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

    def test_dataset_save_and_load_without_preprocessing_options(self):
        # Save the dataset to disk
        self.dataset.save('test_dataset', SAVED_OBJECTS_PATH)

        # Load the dataset from disk
        dataset_from_disk = TwentyNewsGroupsDataset.load('test_dataset', SAVED_OBJECTS_PATH)

        # Remove the dataset previously stored on disk
        rmtree(join_paths(SAVED_OBJECTS_PATH, 'test_dataset'))

        # Check that the original dataset and the dataset saved and loaded are the same
        self.assertEqual(self.dataset, dataset_from_disk)

    def test_dataset_save_and_load_with_preprocessing_options(self):
        trigrams_func = load_func_from_disk('trigrams_func', SAVED_FUNCS_PATH)
        options = DatasetPreprocessingOptions(normalize=True, lowercase=True, stopwords=False, contractions=False,
                                              vulgar_words=True, emails=True, punctuation=False, ngrams='tri',
                                              ngrams_model_func=trigrams_func, lemmatize=True, stem=True,
                                              apostrophes=True, chars=True)
        dataset = deepcopy(self.dataset)
        dataset.preprocessing_options = options

        # Save the dataset to disk
        dataset.save('test_dataset', SAVED_OBJECTS_PATH)

        # Load the dataset from disk
        dataset_from_disk = TwentyNewsGroupsDataset.load('test_dataset', SAVED_OBJECTS_PATH)

        # Remove the dataset previously stored on disk
        rmtree(join_paths(SAVED_OBJECTS_PATH, 'test_dataset'))

        # Check that the original dataset and the dataset saved and loaded are the same
        self.assertEqual(dataset, dataset_from_disk)


if __name__ == '__main__':
    unittest.main()
