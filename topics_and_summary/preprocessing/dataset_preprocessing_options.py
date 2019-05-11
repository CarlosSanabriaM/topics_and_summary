import os
from copy import deepcopy
from typing import Callable, Dict, Any

from topics_and_summary.utils import join_paths, save_obj_to_disk, load_obj_from_disk, save_func_to_disk, \
    load_func_from_disk


class DatasetPreprocessingOptions:
    """
    Class that stores the preprocessing options chosen in a preprocessed dataset.

    The purpose of this class is to apply the same preprocessing options (the ones chosen to preprocess the dataset)
    to the texts given to the TopicsModel that uses the preprocessed dataset with this options. \
    Because of that, this class only stores options that can be applied to texts. For example, this class doesn't \
    store the option 'trash_docs', because this option is only used in the preprocess_dataset() function, and \
    it's not used in the preprocess_text() function.

    This class also stores the ngrams_model_func created when the option ngrams of the preprocess_dataset() \
    function is 'bi' or 'tri'.

    An instance of this class must be created at the end of the the preprocess_dataset() function, \
    and stored in the preprocessing_options attribute of the Dataset object that has been preprocessed.
    """

    def __init__(self, normalize: bool, lowercase: bool, stopwords: bool, contractions: bool, vulgar_words: bool,
                 emails: bool, punctuation: bool, ngrams: str, ngrams_model_func: Callable, lemmatize: bool, stem: bool,
                 apostrophes: bool, chars: bool):
        self.normalize = normalize
        self.lowercase = lowercase
        self.stopwords = stopwords
        self.contractions = contractions
        self.vulgar_words = vulgar_words
        self.emails = emails
        self.punctuation = punctuation
        self.ngrams = ngrams
        self.ngrams_model_func = ngrams_model_func
        self.lemmatize = lemmatize
        self.stem = stem
        self.apostrophes = apostrophes
        self.chars = chars

    def as_dict(self) -> Dict[str, Any]:
        """
        Returns the attributes of this object as a dict. This simplifies how the options stored in this \
        DatasetPreprocessingOptions are passed to the preprocess_text() function.

        Instead of passing them like this:

        >>> from topics_and_summary.preprocessing.text import preprocess_text
        >>> text = 'This is a text'
        >>> options: DatasetPreprocessingOptions
        >>> preprocess_text(text, normalize=options.normalize, lowercase=options.lowercase, ...)

        The options should be passed like this:

        >>> preprocess_text(text, **options.as_dict())
        """
        return deepcopy(vars(self))

    def save(self, name: str, folder_path: str = None):
        """
        Stores the DatasetPreprocessingOptions object attributes on disk. \
        A folder with same name as the name parameter is created inside the folder_path folder. The folder contains:

        * A file with a dict with all the attributes (except the ngrams_model_func)
        * A file with the ngrams_model_func (even if it's None)

        :param name: Name that will have the folder with the object files.
        :param folder_path: Path of the folder where the DatasetPreprocessingOptions folder will be stored on disk.
        """
        # Create the directory
        files_folder = join_paths(folder_path, name)
        os.mkdir(files_folder)

        # Save the dict with all the attributes except the ngrams_model_func
        options_except_ngrams_model_func = self.as_dict()
        # as_dict() returns a copy of the params, so deleting ngrams_model_func from the dict
        # doesn't delete it from the original object
        del options_except_ngrams_model_func['ngrams_model_func']
        save_obj_to_disk(options_except_ngrams_model_func, name + '_options_except_ngrams_model_func', files_folder)

        # Save the ngrams_model_func
        save_func_to_disk(self.ngrams_model_func, name + '_ngrams_model_func', files_folder)

    @classmethod
    def load(cls, name: str, parent_folder_path: str = None) -> 'DatasetPreprocessingOptions':
        """
        Loads the options of a saved DatasetPreprocessingOptions object stored on disk.

        :param name: Name of the folder that contains the DatasetPreprocessingOptions object files.
        :param parent_folder_path: Path of the folder that contains the folder with the object files.
        :return: The DatasetPreprocessingOptions object loaded from disk.
        """
        files_folder = join_paths(parent_folder_path, name)

        # Load all the attributes except the ngrams_model_func (it's a dict)
        # noinspection PyTypeChecker
        options_except_ngrams_model_func: dict = load_obj_from_disk(name + '_options_except_ngrams_model_func',
                                                                    files_folder)

        # Load the ngrams_model_func
        ngrams_model_func = load_func_from_disk(name + '_ngrams_model_func', files_folder)

        # Join them in the same dict
        options = options_except_ngrams_model_func
        options['ngrams_model_func'] = ngrams_model_func

        # Create an instance of this class using the dict
        return cls(**options)

    def __str__(self):
        result = 'DatasetPreprocessingOptions object values:\n'
        for opt, value in self.as_dict().items():
            result += '\t{0}: {1}\n'.format(opt, value)

        return result

    def __eq__(self, other: object) -> bool:
        """
        **Warning:** This __eq__ doesn't compare the functionality of the ngrams_model_func, because a good comparison
        depends on the ngrams generated for a specific dataset.
        """
        if isinstance(other, self.__class__):
            # ngrams_model_func can be None, so we need to do this if-else to check it's equality
            if (self.ngrams_model_func is None and other.ngrams_model_func is not None) or \
                    (self.ngrams_model_func is not None and other.ngrams_model_func is None):
                return False
            else:
                # ngrams_model_func are not compared
                return self.normalize == other.normalize and \
                       self.lowercase == other.lowercase and \
                       self.stopwords == other.stopwords and \
                       self.contractions == other.contractions and \
                       self.vulgar_words == other.vulgar_words and \
                       self.emails == other.emails and \
                       self.punctuation == other.punctuation and \
                       self.ngrams == other.ngrams and \
                       self.lemmatize == other.lemmatize and \
                       self.stem == other.stem and \
                       self.apostrophes == other.apostrophes and \
                       self.chars == other.chars

        return False
