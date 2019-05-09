import unittest
from shutil import rmtree

from topics_and_summary.datasets.twenty_news_groups import TwentyNewsGroupsDataset
from topics_and_summary.preprocessing.dataset import preprocess_dataset, DatasetPreprocessingOptions
from topics_and_summary.preprocessing.text import preprocess_text
from topics_and_summary.tests.paths import SAVED_FUNCS_PATH, SAVED_OBJECTS_PATH
from topics_and_summary.utils import load_func_from_disk, join_paths


class TestPreprocessing(unittest.TestCase):

    def test_preprocess_text(self):
        text = """
        Windows DOS is a family of disk operating systems, hence the name. It isn't a very good OS. What do u think, user@gmail.com?
        Mike's house is too big! N.A.S.A.
        """
        expected_preprocessed_text = 'window disk_operating_system family disk_operating system good mike house big'

        trigrams_func = load_func_from_disk('trigrams_func', SAVED_FUNCS_PATH)
        preprocessed_text = preprocess_text(text, ngrams='tri', ngrams_model_func=trigrams_func)

        self.assertEqual(expected_preprocessed_text, preprocessed_text)

    def test_preprocess_dataset(self):
        # Load dataset and apply all preprocessing
        dataset = TwentyNewsGroupsDataset()
        preprocessed_dataset, trigrams_func = preprocess_dataset(dataset, ngrams='tri')

        # Obtain the dataset stored in disk (it has been preprocessed too, with the same options)
        expected_preprocessed_dataset = TwentyNewsGroupsDataset.load('trigrams_dataset', SAVED_OBJECTS_PATH)

        self.assertEqual(expected_preprocessed_dataset, preprocessed_dataset)

    def test_dataset_preprocessing_options_as_dict(self):
        trigrams_func = load_func_from_disk('trigrams_func', SAVED_FUNCS_PATH)
        options = DatasetPreprocessingOptions(normalize=True, lowercase=True, stopwords=False, contractions=False,
                                              vulgar_words=True, emails=True, punctuation=False, ngrams='tri',
                                              ngrams_model_func=trigrams_func, lemmatize=True, stem=True,
                                              apostrophes=True, chars=True)
        expected_dict = {
            'normalize': True,
            'lowercase': True,
            'stopwords': False,
            'contractions': False,
            'vulgar_words': True,
            'emails': True,
            'punctuation': False,
            'ngrams': 'tri',
            # ngrams_model_func is not included because it can't be directly compared
            'lemmatize': True,
            'stem': True,
            'apostrophes': True,
            'chars': True
        }

        options_dict = options.as_dict()

        # First, check if the ngrams_model_func behaves as expected
        words_list = ['windows', 'disk', 'operating', 'system']
        expected_ngrams = ['windows', 'disk_operating_system']
        self.assertEqual(expected_ngrams, trigrams_func(words_list))
        self.assertEqual(trigrams_func(words_list), options_dict['ngrams_model_func'](words_list))

        # Remove the ngrams_model_func from the options_dict
        del options_dict['ngrams_model_func']

        # Second, check if the rest of options are the expected
        self.assertEqual(expected_dict, options_dict)

    def test_dataset_preprocessing_options_save_and_load(self):
        trigrams_func = load_func_from_disk('trigrams_func', SAVED_FUNCS_PATH)
        options = DatasetPreprocessingOptions(normalize=True, lowercase=True, stopwords=False, contractions=False,
                                              vulgar_words=True, emails=True, punctuation=False, ngrams='tri',
                                              ngrams_model_func=trigrams_func, lemmatize=True, stem=True,
                                              apostrophes=True, chars=True)

        # Save the options to disk
        options.save('test_options', SAVED_OBJECTS_PATH)

        # Load the options from disk
        options_from_disk = DatasetPreprocessingOptions.load('test_options', SAVED_OBJECTS_PATH)

        # Remove the options previously stored on disk
        rmtree(join_paths(SAVED_OBJECTS_PATH, 'test_options'))

        # Check that the original options and the options saved and loaded are the same
        # This doesn't check that the ngrams_model_func behave the same. Only checks if both are None or not None.
        self.assertEqual(options, options_from_disk)
        # Check that the ngrams_model_func behave the same
        words_list = ['windows', 'disk', 'operating', 'system']
        expected_ngrams = ['windows', 'disk_operating_system']
        self.assertEqual(expected_ngrams, options.ngrams_model_func(words_list))
        self.assertEqual(options.ngrams_model_func(words_list), options_from_disk.ngrams_model_func(words_list))


if __name__ == '__main__':
    unittest.main()
