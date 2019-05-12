import abc
import os
import warnings
from copy import deepcopy
from typing import List, Callable

import pandas as pd

from topics_and_summary.preprocessing.dataset_preprocessing_options import DatasetPreprocessingOptions
from topics_and_summary.utils import join_paths, load_obj_from_disk, save_obj_to_disk, \
    get_abspath_from_project_source_root


def get_file_content(file_path: str, encoding: str = None) -> str:
    """
    Returns the content os the specified file as a string.

    :param file_path: Path of the file to be converted into a string.
    :param encoding: Encoding of the file.
    :return: The content of the file as a string.
    """
    with open(file_path, encoding=encoding) as f:
        return f.read()


class Dataset(metaclass=abc.ABCMeta):
    """Class that represents a dataset."""

    def __init__(self, dataset_path: str, encoding: str):
        """
        :param dataset_path: Path to the dataset.
        :param encoding: Encoding that will be used to load the docs from disk with the open() built-in function.
        """
        self.dataset_path = dataset_path
        self.encoding = encoding
        # This attribute will store the preprocessing options applied with the preprocess_dataset() function
        # noinspection PyTypeChecker
        self.preprocessing_options: DatasetPreprocessingOptions = None

    def get_original_doc_content_from_disk(self, doc: 'Document') -> str:
        """
        Given a Document object, this method return it's content obtained from disk as a str.

        :param doc: Document.
        :return: Content of the given document obtained from disk.
        """
        return get_file_content(
            join_paths(self.dataset_path, doc.get_doc_path_inside_dataset_folder()),
            self.encoding
        )

    @abc.abstractmethod
    def apply_function_to_files(self, func: Callable):
        """
        Applies the given function to each of the text files in the corpus.

        :param func: The function to be applied to each text file.
        """

    @abc.abstractmethod
    def as_dataframe(self) -> pd.DataFrame:
        """Returns the files as a pandas DataFrame."""

    @abc.abstractmethod
    def as_documents_content_list(self, tokenize_words=True) -> List[str]:
        """
        Returns a list of the content of the documents in the dataset.

        :param tokenize_words: If true, each document content is converted into a list of words.
        :return: List of the content of the documents in the dataset.
        """

    @abc.abstractmethod
    def as_documents_obj_list(self, tokenize_words=True) -> List['Document']:
        """
        Returns a list of Document objects of the documents in the dataset.

        :param tokenize_words: If true, each document content is converted into a list of words.
        :return: List of Document objects of the documents in the dataset.
        """

    def save(self, name: str, folder_path: str = None):
        """
        Stores the dataset on disk. Creates a folder that contains the files needed to store \
        the dataset object attributes.

        :param name: Name that will have the dataset folder on disk.
        :param folder_path: Path of the folder where the dataset will be stored on disk.
        """
        # Create the directory where all the files will be saved
        files_folder = join_paths(folder_path, name)
        os.mkdir(files_folder)

        # Create a copy of self
        self_copy = deepcopy(self)
        # Remove the preprocessing_options from the self copy
        del self_copy.preprocessing_options

        # Save the copy of self in a file (the preprocessing_options are not saved because where removed from the copy)
        save_obj_to_disk(self_copy, name + '_except_preprocessing_options', files_folder)

        # Save the preprocessing options (if are not None)
        if self.preprocessing_options is not None:
            self.preprocessing_options.save(name + '_preprocessing_options', files_folder)

    @classmethod
    def load(cls, name: str, parent_dir_path: str = None, dataset_path: str = None) -> 'Dataset':
        """
        Loads a saved dataset object from disk.

        :param name: Name of the folder where the dataset object files are stored.
        :param parent_dir_path: Path to the folder where the dataset object folder is stored on disk.
        :param dataset_path: Path to the folder that contains the original dataset documents.
        :return: The dataset loaded from disk.

        For example, consider the following directory structure:

        * stored-datasets-objects/dataset_obj_1/dataset_obj_1_preprocessing_options/...
        * stored-datasets-objects/dataset_obj_1/dataset_obj_1__except_preprocessing_options.pickle
        * datasets/20_newsgroups

        Where 20_newsgroups contains the original 20_newsgroups dataset documents and dataset_obj_1 contains |
        the files of a previously stored dataset object (with the save() method).

        To load the dataset_obj_1 dataset object that contains a dataset object of the 20 newsgroups dataset, \
        this method should be called this way:

        >>> from topics_and_summary.datasets.common import Dataset
        >>> dataset = Dataset.load('dataset_obj_1', 'path/to/stored-datasets-objects', 'path/to/datasets')
        """
        if parent_dir_path is None:
            parent_dir_path = get_abspath_from_project_source_root('saved-elements/objects')

        files_folder = join_paths(parent_dir_path, name)

        # Load the dataset (except the preprocessing options)
        # noinspection PyTypeChecker
        dataset: Dataset = load_obj_from_disk(name + '_except_preprocessing_options', files_folder)

        # If the <dataset-name>_preprocessing_options folder exists, it means that the preprocessing_options where saved
        # In that case, the preprocessing_options are loaded
        if os.path.exists(join_paths(files_folder, name + '_preprocessing_options')):
            dataset.preprocessing_options = \
                DatasetPreprocessingOptions.load(name + '_preprocessing_options', files_folder)
        else:
            dataset.preprocessing_options = None

        # Update the dataset_path of the object if a value is given
        if dataset_path is not None:
            dataset.dataset_path = dataset_path
        else:
            # If the path to the files of the dataset has changed after the dataset object was stored,
            # the dataset_path attribute of the loaded object is wrong, but in this class we don't know the current
            # path of the dataset files, so the user needs to check if the path is ok or it needs to be updated
            warnings.warn("\nThe dataset_path attribute of the loaded dataset object may need to be updated. "
                          "It's current value is: {0}.\n"
                          "If the path to the files of the dataset has changed after the dataset object was stored, "
                          "the dataset_path attribute of the loaded object is wrong and needs to be changed manually.\n"
                          "There are 2 ways to update the dataset path:\n"
                          "\t1. Change it directly in the loaded model: dataset_obj.dataset_path = <path>\n"
                          "\t2. Load the dataset again with load(), specifying the path in the dataset_path parameter"
                          .format(dataset.dataset_path))

        return dataset

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.as_documents_obj_list() == other.as_documents_obj_list() and \
                   self.preprocessing_options == other.preprocessing_options
        return False


class Document(metaclass=abc.ABCMeta):
    """
    Class that represents a dataset document.
    Stores the info needed to access that doc in the dataset directory and the document content.
    """

    def __init__(self, name: str, content: str = None):
        """
        :param name: Name of the document.
        :param content: Content of the document.
        """
        self.name = name
        self.content = content

    @abc.abstractmethod
    def get_doc_path_inside_dataset_folder(self) -> str:
        """Returns the relative path of the document inside the dataset directory."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.content == other.content
        return False
