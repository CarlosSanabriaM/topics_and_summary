import re
import json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from preprocessing.ngrams import make_bigrams_and_get_bigram_model_func, make_trigrams_and_get_trigram_model_func
from utils import get_abspath, pretty_print

__BASIC_STOPWORDS = set(stopwords.words('english'))
__EMAILS_RE = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
__PUNCTUATION_RE = re.compile('[ºª#$€%&*+-_.·,;:<=>@/¡!¿?^¨`´\"(){|}~[\\]]')
__PREPROCESSING_FILES_DIR = '../preprocessing-files/'
__ADDITIONAL_STOPWORDS_PATH = __PREPROCESSING_FILES_DIR + 'stopwords.txt'
__EXPAND_CONTRACTIONS_DICT_PATH = __PREPROCESSING_FILES_DIR + 'expand_contractions_dict.txt'
__VULGAR_WORDS_DICT_PATH = __PREPROCESSING_FILES_DIR + 'vulgar_words_dict.txt'
__NORMALIZE_WORDS_DICT_PATH = __PREPROCESSING_FILES_DIR + 'normalize_words_dict.txt'


def to_lowercase(text):
    """
    Returns the given text with all characters in lowercase.
    :param text: The text to be converted to lowercase. (String)
    :return: The given text with all characters in lowercase. (String)
    """
    text = text.lower()
    return text


def remove_stopwords(text, basic_stopwords=__BASIC_STOPWORDS, additional_stopwords=True):
    """
    Returns the given text with the stopwords removed.
    :param text: The text to remove stopwords. (String)
    :param basic_stopwords: Set of basic stopwords to be removed.
    :param additional_stopwords: Remove additional stopwords (stored in the file __ADDITIONAL_STOPWORDS_PATH).
    By default is true.
    :return: The given text with all the stopwords removed. (String)
    """
    _stopwords = basic_stopwords

    if additional_stopwords:
        additional_stopwords_path = get_abspath(__file__, __ADDITIONAL_STOPWORDS_PATH)
        with open(additional_stopwords_path) as f:
            additional_stopwords = set(line.strip() for line in f)
        _stopwords = _stopwords.union(additional_stopwords)

    return ' '.join(word for word in text.split()
                    if word not in _stopwords)


def remove_emails(text, emails=__EMAILS_RE):
    """
    Returns the given text with the emails removed.
    :param text:
    :param emails:
    :return:
    """
    return emails.sub('', text)


def substitute_punctuation(text, punctuation=__PUNCTUATION_RE, substitute_by=' '):
    """
    Substitutes the punctuation of the given text by the specified string.
    :param text: The text where you want to substitute the punctuation.
    :param punctuation: Regex of the punctuation elements you want to substitute.
    :param substitute_by: The string yo want to use to substitute the punctuation.
    :return: The text with the punctuation substituted.
    """
    return punctuation.sub(substitute_by, text)


def stem_words(text, stemmer=PorterStemmer()):
    """
    Given a string, it applies the specified Stemmer to each word of the string.
    :param text: The text to be stemmed. (String)
    :param stemmer: The stemmer to be applied. The default one is the Porter Stemmer.
    :return: The text stemmed. (String)
    """
    return ' '.join(stemmer.stem(word) for word in text.split())


def lemmatize_words(text, lemmatizer=WordNetLemmatizer()):
    """
    Given a string, it applies the specified lemmatizer to each word of the string.
    :param text: The text to be lemmatized. (String)
    :param lemmatizer: The lemmatizer to be applied. The default one is the WordNet Lemmatizer.
    :return: The text lemmatized. (String)
    """
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split())


def get_first_k_words(text, num_words):
    """
    Returns the given text with only the first k words.
    if k >= len(text), returns the entire text.
    :param text: The text to be limited in the number of words (String).
    :param num_words: The value of k (int). Should be >= 0.
    :return: The given text with only the first k words.
    """
    words = text.split()
    if num_words >= len(text):
        return text

    return ' '.join(words[:num_words])


def substitute_word(text, word, substitute_by):
    """
    Substitutes in the given text the specified word by the substitute_by word.
    It doesn't substitutes part of words. For example, substitute_word('you about u', 'u', '*')
    returns 'you about *' and NOT 'yo* abo*t *'.
    :param text: Text where you want to substitute the word.
    :param word: Word to be substituted.
    :param substitute_by: Word that substitutes the word to be substituted.
    :return: The text with all occurrences of 'word' substituted by 'substitute_by'.
    """
    return re.sub('\\b' + word + '\\b', substitute_by, text)


def substitute_words_with_dict(text, substitutions_dict):
    """
    Substitutes in the given text the elements specified in the substitutions_dict.
    :param text: Text where you want to substitute words.
    :param substitutions_dict: It can be a dict or a str (which corresponds to the path
    to a file containing the dict in json format). The keys of the dict are the words to
    be replaced and the values are the corresponding word that substitutes the word to be replaced.
    If it's a str, the path should be relative to this file.
    :return: The text with the words replaced.
    """

    if not ((type(substitutions_dict) is str) or
            (type(substitutions_dict) is dict)):
        raise TypeError('The substitutions_dict type must be dict or str. '
                        'If it is str, it has to be the path to the dict stored in json format.')

    # If substitutions_dict is a str, then it's the path to a file containing the dict in json format.
    # So, we have to load the dict from the hard drive to memory.
    if type(substitutions_dict) is str:
        file_path = get_abspath(__file__, substitutions_dict)
        with open(file_path) as f:
            substitutions_dict = json.load(f)

    # If it's a dict (it was at the beginning or it was converted above) we make the substitutions.
    for word, substitute_by in substitutions_dict.items():
        text = substitute_word(text, word, substitute_by)

    return text


def expand_contractions(text, expand_contractions_dict=None):
    """
    Expands the contractions (for example: "don't") in the documents of the dataset.
    For example, if expand_contractions_dict has the following pair: "don't": "do not"
    all the appearances in the dataset of the word "don't" will be replaces by "do not".
    :param text: Text where you want to expand the contractions.
    :param expand_contractions_dict: A dict where the keys are the words to be replaced
    and the values the corresponding word that substitutes the word to be replaced.
    If no value is passed, a dict is loaded from the file specified in __EXPAND_CONTRACTIONS_DICT_PATH.
    The dict should be in lowercase.
    :return: The text with the contractions expanded.
    """
    if expand_contractions_dict is None:
        return substitute_words_with_dict(text, __EXPAND_CONTRACTIONS_DICT_PATH)

    return substitute_words_with_dict(text, expand_contractions_dict)


def substitute_vulgar_words(text, vulgar_words_dict=None):
    """
    Substitutes the vulgar words for their normal version. For example: "u" --> "you".
    :param text: Text where you want to substitute the vulgar words.
    :param vulgar_words_dict: A dict where the keys are the words to be replaced
    and the values the corresponding word that substitutes the word to be replaced.
    If no value is passed, a dict is loaded from the file specified in __VULGAR_WORDS_DICT_PATH.
    The dict should be in lowercase.
    :return: The text with the vulgar words substituted.
    """
    if vulgar_words_dict is None:
        return substitute_words_with_dict(text, __VULGAR_WORDS_DICT_PATH)

    return substitute_words_with_dict(text, vulgar_words_dict)


def normalize_words(text, normalize_words_dict=None):
    """
    Normalization/canonicalization is a process for converting data that has more than one possible
    representation into a "standard", "normal", or canonical form.
    A single normalized form is chosen for words with multiple forms like USA and US or uh-huh and uhhuh.
    This function substitutes the words specified in the given dict with the value specified in that dict.
    :param text: Text where you want to substitute some words.
    :param normalize_words_dict: A dict where the keys are the words to be replaced
    and the values the corresponding word that substitutes the word to be replaced.
    If no value is passed, a dict is loaded from the file specified in __NORMALIZE_WORDS_DICT_PATH.
    The dict can have uppercase letter.
    This function should be used before any other preprocessing, for example, convert words to lowercase.
    :return: The text with the words substituted.
    """
    if normalize_words_dict is None:
        return substitute_words_with_dict(text, __NORMALIZE_WORDS_DICT_PATH)

    return substitute_words_with_dict(text, normalize_words_dict)


def remove_single_chars(text):
    """
    Returns the given text with the words that are simple chars (have length 1) removed.
    :param text: The text to remove single chars. (String)
    :return: The given text with all the single chars removed. (String)
    """
    return ' '.join(word for word in text.split()
                    if len(word) > 1)


def preprocess_text(text, normalize=True, lowercase=True, contractions=True, vulgar_words=True,
                    stopwords=True, emails=True, punctuation=True, ngrams='uni', ngrams_model_func=None,
                    lemmatize=True, stem=False, chars=True):
    """
    Receives a str containing a text and returns a list of words after applying the specified preprocessing.
    The original dataset is not modified.
    :type text: str
    :param text: Text to be preprocessed.
    :param normalize: Normalize words. By default is True.
    :param lowercase: Transform to lowercase. By default is True.
    :param contractions: Expand contractions. By default is True.
    :param vulgar_words: Substitute vulgar words. By default is True.
    :param stopwords: Remove stopwords. By default is True.
    :param emails: Remove emails. By default is True.
    :param punctuation: Remove punctuation. By default is True.
    :param ngrams: If 'uni' uses unigrams. If 'bi' create bigrams. If 'tri' creates trigrams. By default is 'uni'.
    If is 'bi' or 'tri', it uses the ngrams_model_func for creating the bi/trigrams.
    :param ngrams_model_func: Function that receives a list of words and returns a list of words with
    possible bigrams/trigrams, based on the bigram/trigram model trained in the given dataset. This function
    is returned by make_bigrams_and_get_bigram_model_func() or make_trigrams_and_get_trigram_model_func() functions in
    the preprocessing.ngrams module. If ngrams is 'uni' this function is not used.
    :param lemmatize: Lemmatize words. By default is True.
    :param stem: Stemm words. By default is False.
    :param chars: Remove single chars. By default is True.
    :return: List of str.
    Note that lemmatize and stem shouldn't be both True.
    """

    # TODO: Revise order of functions
    if normalize:
        # TODO: Problem: Here we can have 'USA,' and 'USA' doesn't detect that.
        text = normalize_words(text)
    if lowercase:
        text = to_lowercase(text)
    if stopwords:  # TODO: Remove also STOPWORDS here??
        text = remove_stopwords(text)
    if contractions:  # TODO: Some of the keys in the file are also in the STOPWORDS. What to do??
        text = expand_contractions(text)
    if vulgar_words:  # TODO: remove this and put the words in that dict in 'normalize_words_dict.txt'????
        text = substitute_vulgar_words(text)
    if emails:
        text = remove_emails(text)
    if punctuation:
        text = substitute_punctuation(text)
    if stopwords:
        text = remove_stopwords(text)
    if ngrams == 'bi' or ngrams == 'tri':
        text = ' '.join(ngrams_model_func(text.split()))
    if lemmatize:
        text = lemmatize_words(text)
    elif stem:
        text = stem_words(text)
    # TODO: remove apostrophes here?? we have things like god's
    if chars:
        text = remove_single_chars(text)

    # TODO: Posible stopwords: people, thing, time, mr, de, st, make
    # TODO: Problems with removing '.'  For example: 'A.M.O.R.C' converts in 'a m o r c'. Try to maintain that letter together and removing '.'.
    # TODO: Windows lemmatizes to window
    # TODO: Where are the numbers removed??

    return text
