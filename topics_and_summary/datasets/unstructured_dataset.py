from copy import deepcopy
from os import listdir
from typing import List, Callable

import pandas as pd

from topics_and_summary.datasets.common import get_file_content, Document, Dataset
from topics_and_summary.utils import join_paths


class UnstructuredDataset(Dataset):
    """
    Class that represents a dataset that is NOT structured in folders.
    All the documents are stored in the dataset folder.
    """

    def __init__(self, dataset_path: str, encoding: str):
        """
        :param dataset_path: Path to the dataset.
        :param encoding: Encoding that will be used to load the docs from disk with the open() built-in function.
        """
        super().__init__(dataset_path, encoding)

        self.files_list: List['Document'] = []  # List of document objects
        self._load_files()  # Load the files in the previous list

    def _load_files(self):
        """
        Load the files in the files_list.
        """
        for file_name in sorted(listdir(self.dataset_path)):
            # Skip hidden files
            if file_name.startswith('.'):
                continue

            file_content = get_file_content(
                join_paths(self.dataset_path, file_name),
                self.encoding
            )

            self.files_list.append(UnstructuredDocument(file_name, file_content))

    def apply_function_to_files(self, func: Callable):
        """
        Applies the given function to each of the text files in the corpus.

        :param func: The function to be applied to each text file.
        """
        for file in self.files_list:
            file.content = func(file.content)

    def remove_document(self, index: int):
        """
        Removes from the dataset the document with the given index.

        :param index: Index of the document inside self.files_list.
        """
        del self.files_list[index]

    def as_dataframe(self) -> pd.DataFrame:
        """Returns the files_list as a pandas DataFrame."""
        i = 0
        dataframe_dict = {}
        for file in self.files_list:
            dataframe_dict[i] = [file.content, file.name]
            i += 1

        return pd.DataFrame.from_dict(dataframe_dict, orient='index', columns=['document', 'document_name'])

    def as_documents_content_list(self, tokenize_words=True) -> List[str]:
        if tokenize_words:
            return [file.content.split() for file in self.files_list]
        return [file.content for file in self.files_list]

    def as_documents_obj_list(self, tokenize_words=True) -> List['Document']:
        if tokenize_words:
            documents_obj_list = []
            files_list = deepcopy(self.files_list)
            for file in files_list:
                file.content = file.content.split()
                documents_obj_list.append(file)
            return documents_obj_list

        return [file for file in self.files_list]

    def get_document_index(self, doc_name: str) -> int:
        """
        Returns the index of the specified document.

        :param doc_name: Name of the document.
        :return: Returns the index of the specified document or -1 if the documents isn't in the dataset.
        """
        try:
            # Get the document with that name
            f = filter(lambda _doc: _doc.name == doc_name, self.files_list)
            doc = next(f)
        except StopIteration:
            # If the filter has 0 elements and we use next(), StopIteration is raised, so we return -1
            return -1

        # If the document exists, we return it's index
        return self.files_list.index(doc)

    # load() method is inherited from Dataset

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.files_list == other.files_list
        return False


class UnstructuredDocument(Document):
    """
    Class that represents a document stored directly inside the dataset folder.
    """

    def __init__(self, name: str, content: str = None):
        """
        :param name: Name of the document.
        :param content: Content of the document.
        """
        super().__init__(name, content)

    def get_doc_path_inside_dataset_folder(self) -> str:
        return self.name

    def __str__(self):
        return 'Name: {0}\n' \
               'Content:\n' \
               '{1}'.format(self.name, self.content)
