import unittest

from datasets.twenty_news_groups import TwentyNewsGroupsDataset
from preprocessing.dataset import preprocess_dataset
from preprocessing.text import preprocess_text
from utils import load_obj_from_disk, get_abspath_from_project_root, join_paths, load_func_from_disk


class TestPreprocessing(unittest.TestCase):
    TESTS_BASE_PATH = get_abspath_from_project_root('tests')

    def test_preprocess_text(self):
        text = """
        Windows DOS is a family of disk operating systems, hence the name. It isn't a very good OS. What do u think, user@gmail.com?
        Mike's house is too big! N.A.S.A.
        """
        expected_preprocessed_text = 'window disk_operating_system family disk_operating system good mike house big'

        trigrams_func = load_func_from_disk('trigrams_func',
                                            join_paths(self.TESTS_BASE_PATH, 'saved-elements/funcs'))
        preprocessed_text = preprocess_text(text, ngrams='tri', ngrams_model_func=trigrams_func)

        self.assertEqual(expected_preprocessed_text, preprocessed_text)

    def test_preprocess_dataset(self):
        # Load dataset and apply all preprocessing
        dataset = TwentyNewsGroupsDataset()
        preprocessed_dataset, trigrams_func = preprocess_dataset(dataset, ngrams='tri')

        # Obtain the dataset stored in disk (it has been preprocessed too, with the same options)
        expected_preprocessed_dataset = load_obj_from_disk('trigrams_dataset',
                                                           join_paths(self.TESTS_BASE_PATH, 'saved-elements/objects'))

        self.assertEqual(expected_preprocessed_dataset, preprocessed_dataset)


if __name__ == '__main__':
    unittest.main()
