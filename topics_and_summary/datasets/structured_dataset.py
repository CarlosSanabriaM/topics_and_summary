import abc
from collections import OrderedDict
from copy import deepcopy
from os import listdir
from typing import List, Callable

import pandas as pd

from topics_and_summary.datasets.common import get_file_content, Document, Dataset
from topics_and_summary.utils import join_paths


class StructuredDataset(Dataset):
    """
    Class that represents a dataset structured in folders.
    Each folder contains an specific set of related documents.
    """

    def __init__(self, dataset_path: str, encoding: str):
        """
        :param dataset_path: Path to the dataset.
        :param encoding: Encoding that will be used to load the docs from disk with the open() built-in function.
        """
        super().__init__(dataset_path, encoding)

        self.files_dict = OrderedDict()  # Key is the parent folder and value of each key is a list of document objects
        self._load_files()  # Load the files in the previous dict
        self.classes = list(self.files_dict.keys())
        self.num_classes = len(self.files_dict)

    def _load_files(self):
        """
        Load the files in the files_dict with the keys being the category of the files, \
        and the values being a list of document objects, where each document is a file of that category.
        """
        for directory in sorted(listdir(self.dataset_path)):
            # Skip hidden files
            if directory.startswith('.'):
                continue

            self.files_dict[directory] = []

            # Add each file in the category to the dict
            for file_name in sorted(listdir(join_paths(self.dataset_path, directory))):
                # Skip hidden files
                if file_name.startswith('.'):
                    continue

                file_content = get_file_content(
                    join_paths(self.dataset_path, directory, file_name),
                    self.encoding
                )

                self.files_dict[directory].append(self._create_structured_document(directory, file_name, file_content))

    def _create_structured_document(self, directory: str, file_name: str, file_content: str) -> 'StructuredDocument':
        """
        Factory Method design pattern. The subclasses override this method, \
        creating and returning the specific structured document that the subclasses represent.

        :param directory_name: Name of the directory this document is stored in. \
        The directory is inside the dataset directory.
        :param name: Name of the document.
        :param content: Content of the document.
        """
        return StructuredDocument(directory, file_name, file_content)

    def apply_function_to_files(self, func: Callable):
        """
        Applies the given function to each of the text files in the corpus.

        :param func: The function to be applied to each text file.
        """
        for category in self.files_dict:
            for file in self.files_dict[category]:
                file.content = func(file.content)

    def remove_document(self, category: str, index_in_category: int):
        """
        Removes from the dataset the document in the given category with the given index inside that category.

        :param category: Category of the document.
        :param index_in_category: Index of the document inside that category.
        """
        del self.files_dict[category][index_in_category]

    def as_dataframe(self) -> pd.DataFrame:
        """Returns the files_dict as a pandas DataFrame."""
        i = 0
        dataframe_dict = {}
        for category, files_list in self.files_dict.items():
            for file in files_list:
                dataframe_dict[i] = [file.content, category, file.name]
                i += 1

        return pd.DataFrame.from_dict(dataframe_dict, orient='index', columns=['document', 'class', 'document_name'])

    def as_documents_content_list(self, tokenize_words=True) -> List[str]:
        if tokenize_words:
            return [file.content.split() for files_list in list(self.files_dict.values()) for file in files_list]
        return [file.content for files_list in list(self.files_dict.values()) for file in files_list]

    def as_documents_obj_list(self, tokenize_words=True) -> List['Document']:
        if tokenize_words:
            documents_obj_list = []
            files_dict = deepcopy(self.files_dict)
            for files_list in list(files_dict.values()):
                for file in files_list:
                    file.content = file.content.split()
                    documents_obj_list.append(file)
            return documents_obj_list

        return [file for files_list in list(self.files_dict.values()) for file in files_list]

    def get_document_index(self, category: str, doc_name: str) -> int:
        """
        Returns the index of the specified document.

        :param category: Category of the document.
        :param doc_name: Name of the document.
        :return: Returns the index of the specified document or -1 if the documents isn't in the dataset.
        """
        try:
            # Get the document with that name inside that category
            f = filter(lambda doc: doc.name == doc_name, self.files_dict[category])
            doc = next(f)
        except StopIteration:
            # If the filter has 0 elements and we use next(), StopIteration is raised, so we return -1
            return -1

        # If the document exists, we return it's index inside the specified category
        return self.files_dict[category].index(doc)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.files_dict == other.files_dict
        return False


class StructuredDocument(Document):
    """
    Class that represents a document inside a category folder.
    This documents have an additional attribute 'directory_name',
    because documents in this dataset are stored in folders.
    """

    def __init__(self, directory_name: str, name: str, content: str = None):
        """
        :param directory_name: Name of the directory this document is stored in. \
        The directory is inside the dataset directory.
        :param name: Name of the document.
        :param content: Content of the document.
        """
        super().__init__(name, content)
        self.directory_name = directory_name

    def get_doc_path_inside_dataset_folder(self) -> str:
        return join_paths(self.directory_name, self.name)

    def __str__(self):
        return 'Directory: {0}\n' \
               'Name: {1}\n' \
               'Content:\n' \
               '{2}'.format(self.directory_name, self.name, self.content)
