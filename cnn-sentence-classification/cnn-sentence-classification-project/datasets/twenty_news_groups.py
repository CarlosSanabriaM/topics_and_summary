import re
from collections import OrderedDict
from os import listdir

import pandas as pd

from datasets.common import get_file_content, Document
from utils import pretty_print, get_abspath_from_project_root, join_paths


class TwentyNewsGroupsDataset:
    __DATASET_PATH = get_abspath_from_project_root('../../text-preprocessing/20_newsgroups')

    def __init__(self, remove_header=True, remove_footer=True, remove_quotes=True, dataset_path=__DATASET_PATH):
        """
        Loads the 20 newsgroups dataset in a dict.
        It can appy a first preprocessing on the dataset files.
        :param remove_header: If true, it removes the header of all files.
        :param remove_footer: If true, it removes the footer of all files.
        :param remove_quotes: If true, it removes the quotes of all files.
        :param dataset_path: Path to the dataset.
        """
        self.dataset_path = dataset_path
        self.files_dict = OrderedDict()  # Key is the parent folder and value of each key is a list of document objects
        self._load_files()  # Load the files in the previous dict
        self.classes = list(self.files_dict.keys())
        self.num_classes = len(self.files_dict)

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
                    join_paths(self.dataset_path, directory, file_name), 'latin1')
                self.files_dict[directory].append(Document(file_name, file_content))

    def apply_function_to_files(self, func):
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

    __QUOTE_RE = re.compile(
        r'(writes in|writes:|wrote:|says:|said:|^In article|^Quoted from|^\||^>)')

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

    def remove_document(self, category, index_in_category):
        """
        Removes from the dataset the document in the given category with the given index inside that category.
        :param category: Category of the document.
        :param index_in_category: Index of the document inside that category.
        """
        del self.files_dict[category][index_in_category]

    def as_dataframe(self):
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

    def as_documents_list(self, tokenize_words=True):
        """
        Returns a list of the documents in the dataset.
        :param tokenize_words: If true, each document is converted into a list of words.
        :return: List of the documents in the dataset.
        """
        if tokenize_words:
            return [file.content.split() for files_list in list(self.files_dict.values()) for file in files_list]
        return [file.content for files_list in list(self.files_dict.values()) for file in files_list]

    def get_document_index(self, category, doc_name):
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

    def print_some_files(self):
        """
        Prints some text files from the corpus.
        """
        pretty_print('File 1')
        print(self.files_dict['sci.med'][1].content)
        pretty_print('File 2')
        print(self.files_dict['sci.electronics'][0].content)
        pretty_print('File 3')
        print(self.files_dict['rec.autos'][16].content)
