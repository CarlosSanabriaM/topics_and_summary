import unittest

from datasets.twenty_news_groups import TwentyNewsGroupsDataset
from preprocessing.dataset import preprocess_dataset
from utils import load_obj_from_disk, get_abspath_from_project_root, join_paths


class TestPreprocessing(unittest.TestCase):
    TESTS_BASE_PATH = get_abspath_from_project_root('tests')

    def test_preprocess_dataset(self):
        # Load dataset and apply all preprocessing
        dataset = TwentyNewsGroupsDataset()
        preprocessed_dataset, trigrams_func = preprocess_dataset(dataset, ngrams='tri')

        # Obtain the dataset stored in disk (it has been preprocessed too, with the same options)
        preprocessed_dataset_from_disk = load_obj_from_disk('trigrams_dataset',
                                                            join_paths(self.TESTS_BASE_PATH, 'saved-elements/objects'))

        # Check that are equal
        self.assertEqual(preprocessed_dataset, preprocessed_dataset_from_disk)


if __name__ == '__main__':
    unittest.main()
