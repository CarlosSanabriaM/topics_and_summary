import re
import textwrap

from topics_and_summary.datasets.common import Document
from topics_and_summary.datasets.structured_dataset import StructuredDataset, StructuredDocument
from topics_and_summary.utils import pretty_print, get_abspath_from_project_source_root, load_obj_from_disk


class TwentyNewsGroupsDataset(StructuredDataset):
    """
    Class that represents the 20_newsgroups dataset.
    This class can apply a first specific preprocessing on the dataset files.
    """

    _DATASET_PATH = get_abspath_from_project_source_root('../datasets/20_newsgroups')
    _DATASET_ENCODING = 'latin1'

    def __init__(self, remove_header=True, remove_footer=True, remove_quotes=True, dataset_path=None):
        """
        :param remove_header: If true, it removes the header of all files.
        :param remove_footer: If true, it removes the footer of all files.
        :param remove_quotes: If true, it removes the quotes of all files.
        :param dataset_path: Path to the dataset.
        """
        if dataset_path is None:
            dataset_path = self._DATASET_PATH

        super().__init__(dataset_path, self._DATASET_ENCODING)

        # Store this values, because are needed in get_original_doc_content_from_disk(), which is override in this class
        self.remove_header = remove_header
        self.remove_footer = remove_footer
        self.remove_quotes = remove_quotes

        # Apply a first preprocessing, specific of this concrete dataset
        if remove_header:
            self._strip_header()
        if remove_footer:
            self._strip_footer()
        if remove_quotes:
            self._strip_quotes()

    def _create_structured_document(self, directory: str, file_name: str, file_content: str):
        return TwentyNewsGroupsDocument(directory, file_name, file_content)

    @classmethod
    def __strip_header(cls, file_text):
        """
        Returns the given text file with the header removed. \
        The header is the text before the first blank line.
        """
        _before, _blankline, after = file_text.partition('\n\n')
        return after

    def _strip_header(self):
        """
        Removes the header of all the files in the corpus. \
        The header is the text before the first blank line.
        """
        self.apply_function_to_files(self.__strip_header)

    __QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:|^In article|^Quoted from|^\||^>)')

    @classmethod
    def __strip_quotes(cls, file_text):
        """
        Returns the given text file with the quotes removed: \
        Lines beginning with the quote characters > or | \
        and lines that commonly introduce quoted sections.
        """
        good_lines = [line for line in file_text.split('\n')
                      if not cls.__QUOTE_RE.search(line)]
        return '\n'.join(good_lines)

    def _strip_quotes(self):
        """
        Removes the quotes of all the files in the corpus. \
        Lines beginning with the quote characters > or | \
        and lines that commonly introduce quoted sections.
        """
        self.apply_function_to_files(self.__strip_quotes)

    @classmethod
    def __strip_footer(cls, file_text):
        """
        Returns the file with the signature block removed: \
        We assume that signatures are at the end of the text, \
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
        Removes the footer (signature) of all the files in the corpus. \
        We assume that signatures are at the end of the text, \
        separated by a blank line or a line made of -.
        """
        self.apply_function_to_files(self.__strip_footer)

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

    def print_some_files(self, n=3, print_file_num=True):
        """
        Prints some text files from the corpus. \
        This function can be used to see how the preprocessing affects the dataset documents.
        """
        category_and_name_list = [
            ('comp.sys.ibm.pc.hardware', '60133'),
            ('sci.space', '59848'),
            ('rec.sport.hockey', '52609')
        ]

        if n > len(category_and_name_list):
            n = len(category_and_name_list)

        for i in range(n):
            if print_file_num:
                pretty_print('File {0}'.format(i+1))

            doc_index_inside_category = self.get_document_index(*category_and_name_list[i])
            print(textwrap.fill(
                self.files_dict[category_and_name_list[i][0]][doc_index_inside_category].content,
                width=80
            ))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.files_dict == other.files_dict and \
                   self.remove_header == other.remove_header and \
                   self.remove_footer == other.remove_footer and \
                   self.remove_quotes == other.remove_quotes
        return False

    @classmethod
    def load(cls, name: str, folder_path: str = None) -> 'TwentyNewsGroupsDataset':
        """
        Loads a saved dataset from disk. This function must be used to load datasets,
        instead of utils.the load_obj_from_disk() function, because this function updates
        the current value of the dataset path.

        :param name: Name of the dataset.
        :param folder_path: Path of the folder where the dataset is stored on disk.
        :return: The object loaded from disk.
        """
        dataset = load_obj_from_disk(name, folder_path)
        dataset.dataset_path = cls._DATASET_PATH
        return dataset


class TwentyNewsGroupsDocument(StructuredDocument):
    """
    Class that represents a 20_newsgroups document.
    """

    def __init__(self, directory_name: str, name: str, content: str = None):
        """
        :param directory_name: Name of the directory this document is stored in. \
        The directory is inside the dataset directory.
        :param name: Name of the document.
        :param content: Content of the document.
        """
        super().__init__(directory_name, name, content)
