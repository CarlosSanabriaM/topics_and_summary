import abc

import numpy as np
from gensim.models import KeyedVectors

from topics_and_summary.utils import get_abspath_from_project_source_root, join_paths


class EmbeddingsModel(metaclass=abc.ABCMeta):
    """Interface for embedding's models."""

    @abc.abstractmethod
    def get_word_vector(self, word: str):
        """
        Returns the embedding vector of the given word. If the word is \
        not present in the embedding's model, a default numpy array full \
        of zeros of the same size of the vectors in the embedding's model is returned.

        :param word: Word to be converted as a vector.
        :return: The embedding vector of the given word.
        """

    @abc.abstractmethod
    def get_vectors_dim(self) -> int:
        """
        :return: The dimension of the embeddings vectors.
        """


class Word2VecModel(EmbeddingsModel):
    """Word2Vec word embedding model."""

    _WORD2VEC_VECTORS_DIM = 300
    _MODEL_PATH = get_abspath_from_project_source_root('../embeddings/word2vec/GoogleNews-vectors-negative{}.bin.gz'
                                                       .format(_WORD2VEC_VECTORS_DIM))

    def __init__(self, model_path: str = None, vectors_dim=_WORD2VEC_VECTORS_DIM):
        """
        Constructs a Word2Vec model using the gensim library.

        :param model_path: The path of the binary word2vec model.
        :param vectors_dim: Dimension of the vectors of the word2vec model. \
        Default is the 100 billion words model pretrained on Google News.
        """
        if model_path is None:
            model_path = self._MODEL_PATH

        self.vectors_dim = vectors_dim
        self.w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    def get_word_vector(self, word: str):
        """
        Returns the word2vec vector of the given word. If the word is not present \
        in the word2vec model, a default numpy array full of zeros of the same size \
        of the vectors in the word2vec model is returned.

        :param word: Word to be converted as a vector.
        :return: The word2vec vector of the given word.
        """
        try:
            return self.w2v_model.word_vec(word, use_norm=False)
        except KeyError:
            return np.zeros(self.vectors_dim)

    def get_word_index(self, word: str) -> int:
        """
        Returns the index of the specified word.

        :param word: Word.
        :return: Index of the specified word. If the word isn't in the vocabulary of the model, \
        returns the index of the UNK token.
        """
        if self.w2v_model.vocab.get(word) is None:
            return self.w2v_model.vocab.get('UNK').index
        return self.w2v_model.vocab.get(word).index

    def get_vectors_dim(self) -> int:
        return self.vectors_dim


class Glove(EmbeddingsModel):
    """Glove word embedding model."""

    _GLOVE_DIR = get_abspath_from_project_source_root('../embeddings/glove/glove.6B')
    _GLOVE_VECTORS_DIM = 100

    def __init__(self, vectors_dim=_GLOVE_VECTORS_DIM, glove_dir: str = None):
        """
        Reads a glove file where contains in each row, in the first position the word, \
        and in the rest of the line the elements of the word vector.

        :param glove_dir: Path where the glove directory is located. That directory must contain \
        text files with the structure mentioned above.
        :param vectors_dim: Size of the word vector. Possible values are: 50, 100, 200, 300.
        """
        if glove_dir is None:
            glove_dir = self._GLOVE_DIR

        self.vectors_dim = vectors_dim
        # A dict where keys are words and values are their corresponding word vectors
        self.embeddings = {}

        with open(join_paths(glove_dir, 'glove.6B.' + str(vectors_dim) + 'd.txt')) as f:
            for line in f:
                # Each line contains: word number_of_the_word_vector.
                # P. e. the 0.418 0.24968 -0.41242 0.1217 0.34527 ...
                values = line.split()
                word = values[0]  # the word is the first element of the line
                word_vector = np.asarray(values[1:], dtype='float32')  # the word vector is the rest
                self.embeddings[word] = word_vector

    def get_word_vector(self, word: str):
        """
        Returns the glove vector of the given word. If the word is not present \
        in the glove model, a default numpy array full of zeros of the same size \
        of the vectors in the glove model is returned.

        :param word: Word to be converted as a vector.
        :return: The glove vector of the given word.
        """
        # If the word is not present in the dict as a key, an array of zeros is returned
        return self.embeddings.get(word, np.zeros(self.vectors_dim))  # second value is the default if key not present

    def get_vectors_dim(self) -> int:
        return self.vectors_dim
