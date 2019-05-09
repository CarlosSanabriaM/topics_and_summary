import os
import re
from copy import deepcopy
from typing import Union, Set, Tuple, List, Callable, Dict, Any

from texttable import Texttable

from topics_and_summary.datasets.common import Dataset
from topics_and_summary.datasets.twenty_news_groups import StructuredDataset
from topics_and_summary.preprocessing.ngrams import make_bigrams_and_get_bigrams_model_func, \
    make_trigrams_and_get_trigrams_model_func
from topics_and_summary.preprocessing.text import to_lowercase, expand_contractions, substitute_vulgar_words, \
    remove_stopwords, substitute_punctuation, lemmatize_words, stem_words, normalize_words, remove_emails, \
    remove_single_chars, remove_apostrophes
from topics_and_summary.utils import join_paths, get_abspath_from_project_source_root, pretty_print, save_obj_to_disk, \
    load_obj_from_disk, save_func_to_disk, load_func_from_disk

_PREPROCESSING_FILES_DIR = get_abspath_from_project_source_root('preprocessing-files')
_TRASH_WORDS_PATH = join_paths(_PREPROCESSING_FILES_DIR, 'trash_words.txt')
_TRASH_DOCS_PATH = join_paths(_PREPROCESSING_FILES_DIR, 'trash_docs.txt')


def print_words_that_contain_elem(dataset: Dataset, elem: str):
    """
    Prints a table with the following info:
        - Word that contains the given element.
        - Number of occurrences of the word in the whole dataset

    :param dataset: Dataset.
    :param elem: Elem contained in the printed words. \
    Will be used to create a regular expression, containing only that elem.
    """
    elem_re = re.compile(elem)

    # Create table for better printing
    table = Texttable()
    table.set_cols_width([30, 10])
    table.set_cols_align(['c', 'c'])

    # Specify header
    table.set_header_align(['c', 'c'])
    table.header(['Word', 'Num occurrences'])

    num_words_contain_elem = 0
    word_occurrence_dict = {}
    for doc in dataset.as_documents_content_list():
        for word in doc:
            if elem_re.search(word) is not None:
                if word not in word_occurrence_dict:
                    word_occurrence_dict[word] = 0
                word_occurrence_dict[word] += 1
                num_words_contain_elem += 1

    # Sort by the number of occurrences and add items to table
    word_occurrence_sorted = sorted(word_occurrence_dict.items(), key=lambda kv: kv[1])
    for word, occurences in word_occurrence_sorted:
        table.add_row([word, occurences])

    print(table.draw())
    print(" Num words with the elem " + elem + ":", num_words_contain_elem)


def print_docs_that_contain_word(dataset: StructuredDataset, word: str, num_chars_preview=70):
    """
    Prints a table with the following properties of all documents in the dataset that contain the given word:
        - Category name
        - Index inside the document list of that category (in the dataset.files_dic)
        - Number of words of that document
        - Number of occurrences of the word in that document
        - Document name in the dataset
        - Preview of the document content

    :param dataset: Dataset.
    :param word: Word contained in the printed documents.
    :param num_chars_preview: Number of characters to show in the preview column.
    """
    # Create table for better printing
    table = Texttable()
    table.set_cols_width([30, 15, 10, 15, 10, num_chars_preview])
    table.set_cols_align(['c', 'c', 'c', 'c', 'c', 'l'])

    # Specify header
    table.set_header_align(['c', 'c', 'c', 'c', 'c', 'c'])
    table.header(['Category name', 'Doc index inside category list of docs',
                  'Num words', 'Num occurrences of given word', 'Document name', 'Content preview'])

    num_docs_contain_word = 0
    for category_name, category_docs in dataset.files_dict.items():
        doc_index_in_category = 0
        for doc in category_docs:
            doc_words = doc.content.split()
            if word in doc_words:
                num_docs_contain_word += 1
                num_words_in_doc = len(doc_words)
                num_word_occurences_in_doc = doc_words.count(word)
                # Add row for each doc that contain the given word
                table.add_row(
                    [category_name, doc_index_in_category,
                     num_words_in_doc, num_word_occurences_in_doc,
                     doc.name,
                     # TODO: Instead of showing the first k characters, it would be better to show text around the word
                     doc.content[:num_chars_preview]])
            doc_index_in_category += 1

    print(table.draw())
    print(" Num docs with the word " + word + ":", num_docs_contain_word)


def print_empty_docs(dataset: StructuredDataset):
    """
    Prints the empty documents in the given dataset.
    """
    # Create table for better printing
    table = Texttable()
    table.set_cols_width([30, 15, 10, 10, 10])
    table.set_cols_align(['c', 'c', 'c', 'c', 'l'])
    # Specify header
    table.set_header_align(['c', 'c', 'c', 'c', 'c'])
    table.header(['Category name', 'Doc index inside category list of docs',
                  'Num words', 'Document name', 'Content preview'])
    num_empty_docs = 0
    for category_name, category_docs in dataset.files_dict.items():
        doc_index_in_category = 0
        for doc in category_docs:
            doc_words = doc.content.split()
            if len(doc_words) == 0:
                num_empty_docs += 1
                num_words_in_doc = len(doc_words)
                # Add row for each doc that contain the given word
                table.add_row(
                    [category_name, doc_index_in_category,
                     num_words_in_doc,
                     doc.name,
                     doc.content])
            doc_index_in_category += 1
    print(table.draw())
    print(" Num empty docs:", num_empty_docs)


def get_docs_that_contain_any_of_the_words(dataset: StructuredDataset, words: Union[str, Set[str]]) \
        -> List[Tuple[str, int]]:
    """
    Returns a list of tuples with the category and index of the docs containing any of the given words.
    Words can be a simple string, representing only one word.

    :param dataset: Dataset.
    :param words: It can be a string representing a single word or a set of strings representing one or more words.
    :return: List of tuples, where tuple[0] contains the category of the doc with the any of the given words \
    and tuple[1] contains the index inside the category of the doc with any of the given words.
    """

    # If words is a str, we convert it to a set
    if type(words) is str:
        words = {words}

    if type(words) is not set:
        raise TypeError('words must be of type str or set.')

    list_docs_with_any_of_the_words = []
    for category_name, category_docs in dataset.files_dict.items():
        doc_index_in_category = 0
        for doc in category_docs:
            doc_words = doc.content.split()
            # If any of the words is in the document, that document is added to the list
            if any(word in doc_words for word in words):
                # Add to the list the docs with any of the words
                list_docs_with_any_of_the_words.append((category_name, doc_index_in_category))
            doc_index_in_category += 1

    return list_docs_with_any_of_the_words


def remove_docs_that_contain_any_of_the_given_words(dataset: StructuredDataset, words: Union[str, Set[str]]):
    """
    Removes from the given dataset the documents that contain one or more of the given words. \
    Words can be a simple string, representing only one word.

    :param dataset: Dataset where docs will be removed. The dataset is modified.
    :param words: It can be a string representing a single word or a set of strings representing one or more words.
    """

    if (type(words) is not str) and (type(words) is not set):
        raise TypeError('words must be of type str or set.')

    docs_with_any_of_the_words = \
        get_docs_that_contain_any_of_the_words(dataset, words)  # List of tuple (category, index_in_category)

    # Indices of the same category must be in descendant order to avoid changing
    # the index of the following docs to delete in that category.

    # First we order the elements on the list, based on the index
    docs_with_any_of_the_words.sort(key=lambda _tuple: _tuple[1], reverse=True)
    # Then we order the elements on the list, based on the category
    docs_with_any_of_the_words.sort(key=lambda _tuple: _tuple[0], reverse=True)

    # For each document that contains any of the words, we delete it
    for category, index_in_category in docs_with_any_of_the_words:
        dataset.remove_document(category, index_in_category)


def remove_docs_that_contain_any_of_the_words_in_file(dataset: StructuredDataset, file_path: str = None):
    """
    Removes from the given dataset the documents that contain one or more of the words in the specified file.

    :param dataset: Dataset where docs will be removed. The dataset is modified.
    :param file_path: Path to the file. The file must contain a word in each line.
    """
    if file_path is None:
        file_path = _TRASH_WORDS_PATH

    with open(file_path) as f:
        words = set(line.strip() for line in f)

    remove_docs_that_contain_any_of_the_given_words(dataset, words)


def remove_trash_docs_specified_in_file(dataset: StructuredDataset, file_path: str = None, file_sep=' '):
    """
    Removes from the given dataset the documents specified in a file.

    :param dataset: Dataset where docs will be removed. The dataset is modified.
    :param file_path: Path to the file where the trash docs are specified. The file must contain in each line \
    the category and then the name of the file to be removed. Category and name must be separated by \
    the element specified in 'file_sep' parameter.
    :param file_sep: Separator of the category and the name in the file.
    """
    if file_path is None:
        file_path = _TRASH_DOCS_PATH

    with open(file_path) as f:
        docs_list = [line.strip().split(file_sep) for line in f]

    for category, doc_name in docs_list:
        # Obtain the index inside the category of the doc with the given name
        index_in_category = dataset.get_document_index(category, doc_name)
        # If the file doesn't exists, we omit it
        if index_in_category == -1:
            continue

        # Remove the document of that category with that index
        dataset.remove_document(category, index_in_category)


def remove_empty_docs(dataset: StructuredDataset) -> int:
    """
    Removes the empty documents of the given dataset.

    :param dataset: Dataset where empty docs will be removed. The dataset is modified.
    :return: The number of documents removed (is the same as the number of empty documents in the dataset).
    """
    num_empty_docs = 0
    for category_name, category_docs in dataset.files_dict.items():
        # Iterate backwards to avoid problems removing while iterating
        doc_index_in_category = len(category_docs) - 1
        for doc in reversed(category_docs):
            doc_words = doc.content.split()
            if len(doc_words) == 0:
                dataset.remove_document(category_name, doc_index_in_category)
                num_empty_docs += 1
            doc_index_in_category -= 1

    return num_empty_docs


def preprocess_dataset(dataset: StructuredDataset, trash_docs=True, normalize=True, lowercase=True, stopwords=True,
                       contractions=True, vulgar_words=True, emails=True, punctuation=True, ngrams='uni',
                       min_bigrams_count=50, bigrams_threshold=75, min_trigrams_count=100, trigrams_threshold=175,
                       lemmatize=True, stem=False, trash_words=True, apostrophes=True, chars=True, empty_docs=True) \
        -> StructuredDataset:
    """
    Creates a copy of the given dataset and returns the copy with the specified preprocessing. \
    The original dataset is not modified.

    :param min_bigrams_count: If ngrams is 'bi' or 'tri', this is the minimum number of occurrences \
    of a bigram to be transformed as a bigram.
    :param bigrams_threshold: If ngrams is 'bi' or 'tri', this is the threshold for creating a bigram.
    :param min_trigrams_count: If ngrams is 'tri', this is the minimum number of occurrences \
    of a trigram to be transformed as a trigram.
    :param trigrams_threshold: If ngrams is 'tri', this is the threshold for creating a trigram.
    :param dataset: Dataset to copy and apply preprocessing.
    :param trash_docs: Remove specified docs. By default is True.
    :param normalize: Normalize words. By default is True.
    :param lowercase: Transform to lowercase. By default is True.
    :param stopwords: Remove stopwords. By default is True.
    :param contractions: Expand contractions. By default is True.
    :param vulgar_words: Substitute vulgar words. By default is True.
    :param emails: Remove emails. By default is True.
    :param punctuation: Remove punctuation. By default is True.
    :param ngrams: If 'uni' uses unigrams. If 'bi' create bigrams and returns bigram function. \
    If 'tri' creates trigrams and returns trigram function. By default is 'uni'.
    :param lemmatize: Lemmatize words. By default is True.
    :param stem: Stemm words. By default is False.
    :param trash_words: Remove documents with any of the 'trash words'. By default is True.
    :param apostrophes: Remove apostrophes.
    :param chars: Remove single chars. By default is True.
    :param empty_docs: Remove empty docs. By default is True.
    :return: The dataset with the preprocessing applied.

    Note that lemmatize and stem shouldn't be both True, because only one of them will be applied.
    """

    # Print the options selected
    pretty_print('Preprocessing the dataset')
    # locals() returns all the local variables in the current function.
    # At the top of the function the only local variables are the parameters to the function.
    params = locals()
    del params['dataset']  # remove the dataset param from the params list, because it's not an option
    print('Options selected:')
    for opt, value in params.items():
        print('\t{0}: {1}'.format(opt, value))

    # Create a copy of the dataset to avoid modifying the given dataset
    dataset_copy = deepcopy(dataset)

    if trash_docs:
        remove_trash_docs_specified_in_file(dataset_copy)
    if normalize:
        # TODO: Problem: Here we can have 'USA,' and the 'USA' in the .txt file doesn't match that.
        # TODO: Problem: It only can transform words, so it can't transform 'United States' to 'USA', i.e.
        dataset_copy.apply_function_to_files(normalize_words)
    if lowercase:
        dataset_copy.apply_function_to_files(to_lowercase)
    if stopwords:
        dataset_copy.apply_function_to_files(remove_stopwords)
    if contractions:
        dataset_copy.apply_function_to_files(expand_contractions)
    if vulgar_words:
        dataset_copy.apply_function_to_files(substitute_vulgar_words)
    if emails:
        dataset_copy.apply_function_to_files(remove_emails)
    if punctuation:
        dataset_copy.apply_function_to_files(substitute_punctuation)
    if stopwords:
        dataset_copy.apply_function_to_files(remove_stopwords)
    if ngrams == 'bi':
        bigrams_model_func = make_bigrams_and_get_bigrams_model_func(dataset_copy, min_bigrams_count, bigrams_threshold)
    elif ngrams == 'tri':
        trigrams_model_func = make_trigrams_and_get_trigrams_model_func(dataset_copy,
                                                                        min_bigrams_count, bigrams_threshold,
                                                                        min_trigrams_count, trigrams_threshold)
    if lemmatize:
        dataset_copy.apply_function_to_files(lemmatize_words)
    elif stem:
        dataset_copy.apply_function_to_files(stem_words)
    if trash_words:
        remove_docs_that_contain_any_of_the_words_in_file(dataset_copy)
    if apostrophes:
        dataset_copy.apply_function_to_files(remove_apostrophes)
    if chars:
        dataset_copy.apply_function_to_files(remove_single_chars)
    if empty_docs:
        remove_empty_docs(dataset_copy)

    # TODO: Change this. ngrams_model_func should be included in the DatasetPreprocessingOptions object
    #  That object must be included inside the dataset_copy object.
    if ngrams == 'bi':
        return dataset_copy, bigrams_model_func
    elif ngrams == 'tri':
        return dataset_copy, trigrams_model_func
    else:
        return dataset_copy


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

    An instance of this class must be created in the preprocess_dataset() function, and stored in the \
    preprocessing_options attribute of the Dataset object that has been preprocessed.
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

        Instead of passing them like this: \
        preprocess_text(text, normalize=obj.normalize, lowercase=obj.lowercase, ...)

        The options should be passed like this: \
        preprocess_text(text, **obj.as_dict())
        """
        return deepcopy(vars(self))

    def save(self, name: str, folder_path: str = None):
        """
        Stores the DatasetPreprocessingOptions object attributes on disk. |
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

    # TODO: This __eq__ doesn't compare the functionality of the ngrams_model_func, because a good comparison
    #  depends on the ngrams generated for a specific dataset.
    def __eq__(self, other: object) -> bool:
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
