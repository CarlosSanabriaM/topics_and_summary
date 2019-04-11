import abc
from typing import List, Callable

import pandas as pd

from utils import join_paths


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

    def __init__(self, dataset_path, encoding):
        """
        :param dataset_path: Path to the dataset.
        :param encoding: Encoding that will be used to load the docs from disk with the open() built-in function.
        """
        self.dataset_path = dataset_path
        self.encoding = encoding

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
        """
        Returns the files as a pandas DataFrame.
        """

    @abc.abstractmethod
    def as_documents_content_list(self, tokenize_words=True):
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

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.as_documents_obj_list() == other.as_documents_obj_list()
        return False

    # TODO: Create a method remove_document()


class Document(metaclass=abc.ABCMeta):
    """
    Class that represents a dataset document.
    Stores the info needed to access that doc in the dataset directory and the document content.
    """

    def __init__(self, name: str, content: str):
        """
        :param name: Name of the document.
        :param content: Content of the document.
        """
        self.name = name
        self.content = content

    @abc.abstractmethod
    def get_doc_path_inside_dataset_folder(self) -> str:
        """
        Returns the relative path of the document inside the dataset directory.
        :return:
        """

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.content == other.content
        return False
