import re
from copy import deepcopy
from typing import Union, Set, Tuple, List

from texttable import Texttable

from topics_and_summary.datasets.common import Dataset
from topics_and_summary.datasets.structured_dataset import StructuredDataset
from topics_and_summary.preprocessing.dataset_preprocessing_options import DatasetPreprocessingOptions
from topics_and_summary.preprocessing.ngrams import make_bigrams_and_get_bigrams_model_func, \
    make_trigrams_and_get_trigrams_model_func
from topics_and_summary.preprocessing.text import to_lowercase, expand_contractions, substitute_vulgar_words, \
    remove_stopwords, substitute_punctuation, lemmatize_words, stem_words, normalize_words, remove_emails, \
    remove_single_chars, remove_apostrophes
from topics_and_summary.utils import join_paths, get_abspath_from_project_source_root, pretty_print

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
        - Index inside the document list of that category (in the dataset.files_dict)
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
    Creates a copy of the given dataset and returns the dataset copy with the specified preprocessing applied. \
    The preprocessing options applied (including the ngrams_model_func if it's the case) are stored in the
    preprocessing_options attribute of the returned dataset. The original dataset is not modified.

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

    ngrams_model_func = None

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
        ngrams_model_func = make_bigrams_and_get_bigrams_model_func(dataset_copy, min_bigrams_count, bigrams_threshold)
    elif ngrams == 'tri':
        ngrams_model_func = make_trigrams_and_get_trigrams_model_func(dataset_copy,
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

    # Store the preprocessing options in the dataset copy object
    dataset_copy.preprocessing_options = DatasetPreprocessingOptions(
        normalize=normalize, lowercase=lowercase, stopwords=stopwords, contractions=contractions,
        vulgar_words=vulgar_words, emails=emails, punctuation=punctuation, ngrams=ngrams,
        ngrams_model_func=ngrams_model_func, lemmatize=lemmatize, stem=stem, apostrophes=apostrophes, chars=chars
    )

    return dataset_copy
