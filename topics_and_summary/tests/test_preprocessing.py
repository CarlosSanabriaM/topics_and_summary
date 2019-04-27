import unittest

from topics_and_summary.datasets.twenty_news_groups import TwentyNewsGroupsDataset
from topics_and_summary.preprocessing.dataset import preprocess_dataset
from topics_and_summary.preprocessing.text import preprocess_text
from topics_and_summary.tests.paths import TESTS_BASE_PATH
from topics_and_summary.utils import load_obj_from_disk, join_paths, load_func_from_disk


class TestPreprocessing(unittest.TestCase):

    def test_preprocess_text(self):
        text = """
        Windows DOS is a family of disk operating systems, hence the name. It isn't a very good OS. What do u think, user@gmail.com?
        Mike's house is too big! N.A.S.A.
        """
        expected_preprocessed_text = 'window disk_operating_system family disk_operating system good mike house big'

        trigrams_func = load_func_from_disk('trigrams_func',
                                            join_paths(TESTS_BASE_PATH, 'saved-elements/funcs'))
        preprocessed_text = preprocess_text(text, ngrams='tri', ngrams_model_func=trigrams_func)

        self.assertEqual(expected_preprocessed_text, preprocessed_text)

    def test_preprocess_dataset(self):
        # Load dataset and apply all preprocessing
        dataset = TwentyNewsGroupsDataset()
        preprocessed_dataset, trigrams_func = preprocess_dataset(dataset, ngrams='tri')

        # Obtain the dataset stored in disk (it has been preprocessed too, with the same options)
        expected_preprocessed_dataset = load_obj_from_disk('trigrams_dataset',
                                                           join_paths(TESTS_BASE_PATH, 'saved-elements/objects'))

        self.assertEqual(expected_preprocessed_dataset, preprocessed_dataset)


if __name__ == '__main__':
    unittest.main()
