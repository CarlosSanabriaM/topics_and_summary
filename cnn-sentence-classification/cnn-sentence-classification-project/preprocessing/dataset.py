from utils import get_abspath
from copy import deepcopy
from texttable import Texttable

from utils import pretty_print
from preprocessing.text import to_lowercase, expand_contractions, substitute_vulgar_words, remove_stopwords, \
    substitute_punctuation, lemmatize_words, stem_words, normalize_words, remove_emails, remove_single_chars
from preprocessing.ngrams import make_bigrams_and_get_bigram_model_func, make_trigrams_and_get_trigram_model_func

__PREPROCESSING_FILES_DIR = '../preprocessing-files/'
__TRASH_WORDS_PATH = __PREPROCESSING_FILES_DIR + 'trash_words.txt'
__TRASH_DOCS_PATH = __PREPROCESSING_FILES_DIR + 'trash_docs.txt'


# TODO: Change to admit more than one word ?? I think it makes more sense to be used only with one word.
def print_docs_that_contain_word(dataset, word, num_chars_preview=70):
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
                     doc.content[:num_chars_preview]])  # TODO: Try to preview parts of the text around the word?
            doc_index_in_category += 1

    print(table.draw())
    print(" Num docs with the word " + word + ":", num_docs_contain_word)


# TODO: Refactor?
def print_empty_docs(dataset):
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


def get_docs_that_contain_any_of_the_words(dataset, words):
    """
    Returns a list of tuples with the category and index of the docs containing any of the given words.
    Words can be a simple string, representing only one word.
    :param dataset: Dataset.
    :param words: It can be:
    - A string representing a single word.
    - A set of strings representing one or more words.
    :return: List of tuples, where tuple[0] contains the category of the doc with the any of the given words
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


def remove_docs_that_contain_any_of_the_given_words(dataset, words):
    """
    Removes from the given dataset the documents that contain one or more of the given words.
    Words can be a simple string, representing only one word.
    :param dataset: Dataset where docs will be removed. The dataset is modified.
    :param words: It can be:
    - A string representing a single word.
    - A set of strings representing one or more words.
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


def remove_docs_that_contain_any_of_the_words_in_file(dataset, file_path=__TRASH_WORDS_PATH):
    """
    Removes from the given dataset the documents that contain one or more of the words in the specified file.
    :param dataset: Dataset where docs will be removed. The dataset is modified.
    :param file_path: Path to the file. The file must contain a word in each line.
    """
    file_path = get_abspath(__file__, file_path)
    with open(file_path) as f:
        words = set(line.strip() for line in f)

    remove_docs_that_contain_any_of_the_given_words(dataset, words)


def remove_trash_docs_specified_in_file(dataset, file_path=__TRASH_DOCS_PATH, file_sep=' '):
    """
    Removes from the given dataset the documents specified in a file.
    :param dataset: Dataset where docs will be removed. The dataset is modified.
    :param file_path: Path to the file. The file must contain in each line the category
    and then the name of the file to be removed. Category and name must be separeted by
    the element specified in 'file_sep' parameter.
    :param file_sep: Separator of the category and the name in the file.
    """
    file_path = get_abspath(__file__, file_path)
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


def remove_empty_docs(dataset):
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


def preprocess_dataset(dataset, normalize=True, lowercase=True, contractions=True, vulgar_words=True,
                       stopwords=True, emails=True, punctuation=True, ngrams='uni',
                       lemmatize=True, stem=False, trash_words=True, trash_docs=True, chars=True, empty_docs=True):
    """
    Creates a copy of the given dataset and returns the copy with the specified preprocessing.
    The original dataset is not modified.
    :param dataset: Dataset to copy and apply preprocessing.
    :param trash_docs: Remove specified docs. By default is True.
    :param normalize: Normalize words. By default is True.
    :param lowercase: Transform to lowercase. By default is True.
    :param contractions: Expand contractions. By default is True.
    :param vulgar_words: Substitute vulgar words. By default is True.
    :param stopwords: Remove stopwords. By default is True.
    :param emails: Remove emails. By default is True.
    :param punctuation: Remove punctuation. By default is True.
    :param ngrams: If 'uni' uses unigrams. If 'bi' create bigrams and returns bigram function.
    If 'tri' creates trigrams and returns trigram function. By default is 'uni'.
    :param lemmatize: Lemmatize words. By default is True.
    :param stem: Stemm words. By default is False.
    :param trash_words: Remove documents with any of the 'trash words'. By default is True.
    :param chars: Remove single chars. By default is True.
    :param empty_docs: Remove empty docs. By default is True.
    :return: The dataset with the preprocessing applied.
    Note that lemmatize and stem shouldn't be both True.
    """
    pretty_print('Preprocessing the dataset')  # TODO: Print the options selected: lowercase, contractions, ...
    dataset_copy = deepcopy(dataset)

    # TODO: Revise order of functions
    if trash_docs:
        remove_trash_docs_specified_in_file(dataset_copy)
    if normalize:
        # TODO: Pasamos de tener 246 apariciones de 'usa' a tener 1231.
        # TODO: Problem: Here we can have 'USA,' and 'USA' doesn't detect that.
        # TODO: Problem: It only can transform words. You can't transform 'United States' to 'USA'
        dataset_copy.apply_function_to_files(normalize_words)
    if lowercase:
        dataset_copy.apply_function_to_files(to_lowercase)
    if stopwords:  # TODO: Remove also STOPWORDS here??
        dataset_copy.apply_function_to_files(remove_stopwords)
    if contractions:  # TODO: Some of the keys in the file are also in the STOPWORDS. What to do??
        dataset_copy.apply_function_to_files(expand_contractions)
    if vulgar_words:  # TODO: remove this and put the words in that dict in 'normalize_words_dict.txt'????
        dataset_copy.apply_function_to_files(substitute_vulgar_words)
    if emails:
        dataset_copy.apply_function_to_files(remove_emails)
    if punctuation:
        dataset_copy.apply_function_to_files(substitute_punctuation)
    if stopwords:
        dataset_copy.apply_function_to_files(remove_stopwords)
    # TODO: Bigrams/trigrams here??
    if ngrams == 'bi':
        bigram_model_func = make_bigrams_and_get_bigram_model_func(dataset_copy)
    elif ngrams == 'tri':
        trigram_model_func = make_trigrams_and_get_trigram_model_func(dataset_copy)
    if lemmatize:
        dataset_copy.apply_function_to_files(lemmatize_words)
    elif stem:
        dataset_copy.apply_function_to_files(stem_words)
    if trash_words:
        remove_docs_that_contain_any_of_the_words_in_file(dataset_copy)
    # TODO: remove apostrophes here?? we have things like god's
    if chars:
        dataset_copy.apply_function_to_files(remove_single_chars)
    if empty_docs:
        remove_empty_docs(dataset_copy)

    # TODO: Posible stopwords: people, thing, time, mr, de, st, make

    # TODO: Problems with removing '.'  For example: 'A.M.O.R.C' converts in 'a m o r c'. Try to maintain that letter together and removing '.'.
    # TODO: Windows lemmatizes to window
    # TODO: Where are the numbers removed??

    if ngrams == 'bi':
        return dataset_copy, bigram_model_func
    elif ngrams == 'tri':
        return dataset_copy, trigram_model_func
    else:
        return dataset_copy
