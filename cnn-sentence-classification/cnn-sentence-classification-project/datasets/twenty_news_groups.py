import re
from collections import OrderedDict
from copy import deepcopy
from os import listdir
from typing import List, Callable

import pandas as pd

from datasets.common import get_file_content, Document, Dataset
from utils import pretty_print, get_abspath_from_project_root, join_paths


class TwentyNewsGroupsDataset(Dataset):
    """
    Class that represents the 20_newsgroups dataset.
    This class can apply a first specific preprocessing on the dataset files.
    """

    __DATASET_PATH = get_abspath_from_project_root('../../text-preprocessing/20_newsgroups')
    __DATASET_ENCODING = 'latin1'

    def __init__(self, remove_header=True, remove_footer=True, remove_quotes=True, dataset_path=__DATASET_PATH):
        """
        :param remove_header: If true, it removes the header of all files.
        :param remove_footer: If true, it removes the footer of all files.
        :param remove_quotes: If true, it removes the quotes of all files.
        :param dataset_path: Path to the dataset.
        """
        super().__init__(dataset_path, self.__DATASET_ENCODING)

        self.files_dict = OrderedDict()  # Key is the parent folder and value of each key is a list of document objects
        self._load_files()  # Load the files in the previous dict
        self.classes = list(self.files_dict.keys())
        self.num_classes = len(self.files_dict)

        # Store this values, because are needed in get_original_doc_content_from_disk()
        self.remove_header = remove_header
        self.remove_footer = remove_footer
        self.remove_quotes = remove_quotes

        # Apply a first preprocessing
        if remove_header:
            self._strip_header()
        if remove_footer:
            self._strip_footer()
        if remove_quotes:
            self._strip_quotes()

    def _load_files(self):
        """
        Load the files in the files_dict with the keys being the category of the files,
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
                    self.__DATASET_ENCODING
                )

                self.files_dict[directory].append(TwentyNewsGroupsDocument(directory, file_name, file_content))

    def apply_function_to_files(self, func: Callable):
        """
        Applies the given function to each of the text files in the corpus.
        :param func: The function to be applied to each text file.
        """
        for category in self.files_dict:
            for file in self.files_dict[category]:
                file.content = func(file.content)

    @classmethod
    def __strip_header(cls, file_text):
        """
        Returns the given text file with the header removed.
        The header is the text before the first blank line.
        """
        _before, _blankline, after = file_text.partition('\n\n')
        return after

    def _strip_header(self):
        """
        Removes the header of all the files in the corpus.
        The header is the text before the first blank line.
        """
        self.apply_function_to_files(self.__strip_header)

    __QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:|^In article|^Quoted from|^\||^>)')

    @classmethod
    def __strip_quotes(cls, file_text):
        """
        Returns the given text file with the quotes removed:
        Lines beginning with the quote characters > or |
        and lines that commonly introduce quoted sections.
        """
        good_lines = [line for line in file_text.split('\n')
                      if not cls.__QUOTE_RE.search(line)]
        return '\n'.join(good_lines)

    def _strip_quotes(self):
        """
        Removes the quotes of all the files in the corpus.
        Lines beginning with the quote characters > or |
        and lines that commonly introduce quoted sections.
        """
        self.apply_function_to_files(self.__strip_quotes)

    @classmethod
    def __strip_footer(cls, file_text):
        """
        Returns the file with the signature block removed:
        We assume that signatures are at the end of the text,
        separated by a blank line or a line made of -.
        """
        lines = file_text.strip().split('\n')
        for line_num in range(len(lines) - 1, -1, -1):
            line = lines[line_num]
            if line.strip().strip('-') == '':
                break

        if line_num > 0:
            return '\n'.join(lines[:line_num])
        else:
            return file_text

    def _strip_footer(self):
        """
        Removes the footer (signature) of all the files in the corpus.
        We assume that signatures are at the end of the text,
        separated by a blank line or a line made of -.
        """
        self.apply_function_to_files(self.__strip_footer)

    # TODO: Change to override superclass method (if is created)
    def remove_document(self, category: str, index_in_category: int):
        """
        Removes from the dataset the document in the given category with the given index inside that category.
        :param category: Category of the document.
        :param index_in_category: Index of the document inside that category.
        """
        del self.files_dict[category][index_in_category]

    def as_dataframe(self) -> pd.DataFrame:
        """
        Returns the files_dict as a pandas DataFrame.
        """
        i = 0
        dataframe_dict = {}
        for category, files_list in self.files_dict.items():
            for file in files_list:
                dataframe_dict[i] = [file.content, category, file.name]
                i += 1

        return pd.DataFrame.from_dict(dataframe_dict, orient='index', columns=['document', 'class', 'document_name'])

    def as_documents_content_list(self, tokenize_words=True):
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

    def get_document_index(self, category: str, doc_name: str):
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

    def get_original_doc_content_from_disk(self, doc: 'Document') -> str:
        # The original doc content has the header, footer and quotes
        content = super(TwentyNewsGroupsDataset, self).get_original_doc_content_from_disk(doc)

        # Preprocess
        if self.remove_header:
            content = self.__strip_header(content)
        if self.remove_footer:
            content = self.__strip_footer(content)
        if self.remove_quotes:
            content = self.__strip_quotes(content)

        return content

    def print_some_files(self):
        """
        Prints some text files from the corpus.
        This function can be used to see how the preprocessing affects the dataset documents.
        """
        pretty_print('File 1')
        print(self.files_dict['sci.med'][1].content)
        pretty_print('File 2')
        print(self.files_dict['sci.electronics'][0].content)
        pretty_print('File 3')
        print(self.files_dict['rec.autos'][16].content)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.files_dict == other.files_dict and \
                   self.remove_header == other.remove_header and \
                   self.remove_footer == other.remove_footer and \
                   self.remove_quotes == other.remove_quotes
        return False


class TwentyNewsGroupsDocument(Document):
    """
    Class that represents a 20_newsgroups document.
    Documents of the 20_newsgroups dataset have an additional attribute 'directory_name',
    because documents in this dataset are stored in folders.
    """

    def __init__(self, directory_name: str, name: str, content: str = None):
        """
        :param directory_name: Name of the directory this document is stored in.
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
