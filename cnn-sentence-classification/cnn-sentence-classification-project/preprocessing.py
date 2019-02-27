import re
from nltk.corpus import stopwords  # TODO: nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer  # TODO: nltk.download('wordnet')

__STOPWORDS = set(stopwords.words('english'))
__PUNCTUATION_RE = re.compile('[.,;¡!¿?]')


def to_lowercase_and_remove_stopwords(text, _stopwords=__STOPWORDS):
    """
    Returns the given text with all characters in lowercase and with the stopwords removed.
    :param text: The text to be converted to lowercase an to remove stopwords. (String)
    :param _stopwords: Set of stopwords to be removed.
    :return: The given text with all characters in lowercase and with the stopwords removed. (String)
    """
    text = text.lower()
    text = ' '.join(word for word in text.split()
                    if word not in _stopwords)
    return text


def substitute_punctuation(text, punctuation=__PUNCTUATION_RE, substitute_by=' '):
    """
    Substitutes the punctuation of the given text by the specified string.
    :param text: The text where you want to substitute the puntuation.
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
