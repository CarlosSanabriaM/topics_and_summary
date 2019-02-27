import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from datasets import TwentyNewsGroupsDataset
from preprocessing import get_first_k_words
import preprocessing as pre
from embeddings import Word2VecModel
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical


def preprocess_dataset(dataset):
    """
    Apply all the preprocessing to the given TwentyNewsGroupsDataset dataset.
    """
    print('\n\nPreprocessing the dataset\n_____________________________')

    dataset.apply_function_to_files(pre.to_lowercase_and_remove_stopwords)
    dataset.apply_function_to_files(pre.substitute_punctuation)
    dataset.apply_function_to_files(pre.lemmatize_words)


def prepare_corpus(documents):
    """
    TODO: complete
    :param documents: List of documents. Each document is a list of words, where each word is a string.
    :return:
    """
    """
    Input  : 
    Purpose: create term dictionary of our courpus and Converting list of documents (corpus) into Document Term Matrix
    Output : term dictionary and Document Term Matrix
    """
    # Creating the term dictionary of our courpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(documents)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in documents]
    # generate LDA model
    return dictionary, doc_term_matrix


if __name__ == '__main__':
    # %%
    # Load dataset and apply preprocessing
    dataset = TwentyNewsGroupsDataset()

    # Apply more preprocessing
    preprocess_dataset(dataset)

    # %%
    # Create the term-document matrix and the dictionary of terms
    prepare_corpus(dataset.as_documents_list())
    # TODO: Comprobar que hace el metodo anterior
