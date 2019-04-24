from typing import Tuple, Callable

import gensim

from datasets.common import Dataset


def create_bigram_model(dataset: Dataset, min_count=50, threshold=75) -> gensim.models.phrases.Phraser:
    """
    Returns a bigram model based on the documents in the dataset.
    :param dataset: Dataset.
    :param min_count: Ignore all words and bigrams with total collected count lower than this value.
    :param threshold: Represent a score threshold for forming the phrases (higher means fewer phrases).
    :return: Bigram model.
    """
    bigram = gensim.models.Phrases(dataset.as_documents_content_list(), min_count=min_count, threshold=threshold)

    # Return a Phraser, because it is much smaller and somewhat faster than using the full Phrases model
    return gensim.models.phrases.Phraser(bigram)


def create_trigram_model(dataset: Dataset, min_count1=50, threshold1=75, min_count2=100, threshold2=175) \
        -> Tuple[gensim.models.phrases.Phraser, gensim.models.phrases.Phraser]:
    """
    Returns a bigram and a trigram model based on the documents in the dataset.
    :param dataset: Dataset.
    :param min_count1: Ignore all words and bigrams with total collected count lower than this value
    in the bigram model.
    :param min_count2: Ignore all words and bigrams with total collected count lower than this value
    in the trigram model.
    :param threshold1: Represent a score threshold for forming the phrases for the bigram model
    (higher means fewer phrases).
    :param threshold2: Represent a score threshold for forming the phrases for the trigram model
    (higher means fewer phrases).
    :return: Bigram and Trigram model. Both are needed to create trigrams.
    """
    docs_list = dataset.as_documents_content_list()
    bigram = gensim.models.Phrases(docs_list, min_count=min_count1, threshold=threshold1)
    trigram = gensim.models.Phrases(bigram[docs_list], min_count=min_count2, threshold=threshold2)

    # Return a Phraser, because it is much smaller and somewhat faster than using the full Phrases model
    return gensim.models.phrases.Phraser(bigram), gensim.models.phrases.Phraser(trigram)


def make_ngrams(dataset: Dataset, ngram_model_func: Callable):
    """
    Applies the given ngram model function to the documents in the given dataset.
    :param dataset: Dataset where ngrams will be created.
    :param ngram_model_func: Function where an n-gram model is accesed with the indexing operators [].
    It could be bigram, trigram, ... Example of function: 'lambda x: bigram_model[x]' or
    'lambda x: trigram_model[bigram_model[x]]'.
    """
    dataset.apply_function_to_files(lambda doc: ' '.join(ngram_model_func(doc.split())))


def make_bigrams_and_get_bigram_model_func(dataset: Dataset, min_count=50, threshold=75) -> Callable:
    """
    Creates a bigram model based on the documents in the dataset, makes bigrams in the
    dataset using that model, and returns a function that allows to obtain bigrams
    given a list of strings.
    :param dataset: Dataset to use to create the bigram model. The documents in the dataset
    are modified, creating bigrams with the previously created bigram model.
    :param min_count: Ignore all words and bigrams with total collected count lower than this value.
    :param threshold: Represent a score threshold for forming the phrases (higher means fewer phrases).
    :return: Function that receives a list of words and returns a list of words with
    possible bigrams, based on the bigram model trained in the given dataset.
    """
    bigram_model = create_bigram_model(dataset, min_count=min_count, threshold=threshold)
    bigram_model_func = lambda x: bigram_model[x]
    make_ngrams(dataset, bigram_model_func)

    return bigram_model_func


def make_trigrams_and_get_trigram_model_func(dataset: Dataset, min_count1=50, threshold1=75,
                                             min_count2=100, threshold2=175) -> Callable:
    """
    Creates a trigram model based on the documents in the dataset, makes trigrams in the
    dataset using that model, and returns a function that allows to obtain trigrams
    given a list of strings.
    :param dataset: Dataset to use to create the trigram model. The documents in the dataset
    are modified, creating trigrams with the previously created trigram model.
    :param min_count1: Ignore all words and bigrams with total collected count lower than this value
    in the bigram model.
    :param min_count2: Ignore all words and bigrams with total collected count lower than this value
    in the trigram model.
    :param threshold1: Represent a score threshold for forming the phrases for the bigram model
    (higher means fewer phrases).
    :param threshold2: Represent a score threshold for forming the phrases for the trigram model
    (higher means fewer phrases).
    :return: Function that receives a list of words and returns a list of words with
    possible trigrams, based on the trigram model trained in the given dataset.
    """

    bigram_model, trigram_model = create_trigram_model(dataset, min_count1, threshold1, min_count2, threshold2)
    trigram_model_func = lambda x: trigram_model[bigram_model[x]]
    make_ngrams(dataset, trigram_model_func)

    return trigram_model_func
