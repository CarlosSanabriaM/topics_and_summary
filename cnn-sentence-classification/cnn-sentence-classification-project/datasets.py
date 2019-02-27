import os
import re
import pandas as pd


def get_file_content(file_path, encoding):
    """
    Returns the content os the specified file as a string.
    :param file_path: Path of the file to be converted into a string (String).
    :param encoding: Encoding of the file (String).
    :return: The content of the file as a string.
    """
    with open(file_path, encoding=encoding) as f:
        return f.read()


class TwentyNewsGroupsDataset:
    __DATASET_PATH = '../../text-preprocessing/20_newsgroups'

    def __init__(self, remove_header=True, remove_footer=True, remove_quotes=True, dataset_path=__DATASET_PATH):
        """
        Loads the 20 newsgroups dataset in a dict.
        It can appy a first preprocessing on the dataset files.
        :param remove_header: If true, it removes the header of all files.
        :param remove_footer: If true, it removes the footer of all files.
        :param remove_quotes: If true, it removes the quotes of all files.
        :param dataset_path: Path to the dataset.
        """
        self.dataset_path = dataset_path;
        # The key is the parent folder and the value of each key is
        # a list of strings containing each file in a string format
        self.files_dict = {}
        self._load_files()  # Load the files in the previous dict
        self.classes = list(self.files_dict.keys())
        self.num_classes = len(self.files_dict)

        # Apply a first preprocessing
        if remove_header: self._strip_header()
        if remove_footer: self._strip_footer()
        if remove_quotes: self._strip_quotes()

    def _load_files(self):
        """
        Load the files in the files_dict with the keys being the class to predict,
        and the values being a list of strings, where each string is a file of that class.
        """
        for directory in os.listdir(self.dataset_path):
            # Skip Mac file
            if directory == '.DS_Store':
                continue

            self.files_dict[directory] = []

            for file in os.listdir(self.dataset_path + '/' + directory):
                file_content = get_file_content(
                    self.dataset_path + '/' + directory + '/' + file, 'latin1')
                self.files_dict[directory].append(file_content)

    def apply_function_to_files(self, func):
        """
        Applies the given function to each of the text files in the corpus.
        :param func: The function to be applied to each text file.
        """
        for files_class in self.files_dict:
            self.files_dict[files_class] = [func(file)
                                            for file in self.files_dict[files_class]]

    # TODO: Or @staticmethod??
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

    def as_dataframe(self):
        """
        Returns the files_dict as a pandas DataFrame.
        """
        i = 0
        dataframe_dict = {}
        for file_class, files_list in self.files_dict.items():
            for file in files_list:
                dataframe_dict[i] = [file, file_class]
                i += 1

        return pd.DataFrame.from_dict(dataframe_dict, orient='index', columns=['document', 'class'])

    def as_documents_list(self, tokenize_words=True):
        """
        Returns a list of the documents in the dataset.
        :param tokenize_words: If true, each document is converted into a list of words.
        :return:
        """
        if tokenize_words:
            return [item.split() for sublist in list(self.files_dict.values()) for item in sublist]
        return [item for sublist in list(self.files_dict.values()) for item in sublist]

    def print_some_files(self):
        """
        Prints some text files from the corpus.
        """
        print('File 1\n_______\n')
        print(self.files_dict['sci.med'][1])
        print('\n\nFile 2\n_______\n')
        print(self.files_dict['sci.electronics'][0])
        print('\n\nFile 3\n_______\n')
        print(self.files_dict['rec.autos'][16])
