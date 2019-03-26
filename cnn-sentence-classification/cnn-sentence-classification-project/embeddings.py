import numpy as np
from gensim.models import KeyedVectors

from utils import get_abspath_from_project_root, join_paths


class Word2VecModel:
    """Word2Vec word embedding model."""
    __MODEL_PATH = get_abspath_from_project_root('../../word2vec/GoogleNews-vectors-negative300.bin.gz')

    def __init__(self, model_path=__MODEL_PATH):
        """
        Constructs a Word2Vec model using the gensim library.
        :param model_path: The path of the binary word2vec model.
        Default is the 100 billion words model pretrained on Google News.
        """
        self.w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    def get_word_vector(self, word):
        """
        Returns the word2vec vector of the given word.
        :param word: Word to be converted as a vector.
        :return: The word2vec vector of the given word.
        """
        return self.w2v_model.word_vec(word, use_norm=False)

    def get_word_index(self, word):
        """
        Returns the index of the specified word.
        :param word:
        :return: Index of the specified word. If the word isn't in the vocabulary of the model,
        returns the index of the UNK token.
        """
        if self.w2v_model.vocab.get(word) is None:
            return self.w2v_model.vocab.get('UNK').index
        return self.w2v_model.vocab.get(word).index

    def get_keras_embedding(self, train_embeddings=False):
        """
        :param train_embeddings: If True, the embedding layer weights are updated during training.
        :return: The keras embedding layer using the Word2Vec model.
        """
        return self.w2v_model.get_keras_embedding(train_embeddings=train_embeddings)


class Glove:
    """Glove word embedding model."""
    __GLOVE_DIR = get_abspath_from_project_root('../../glove/glove.6B')
    __GLOVE_VECTORS_DIM = 100

    def __init__(self, vectors_dim=__GLOVE_VECTORS_DIM, glove_dir=__GLOVE_DIR):
        """
        Reads a glove file where contains in each row, in the first position the word,
        and in the rest of the line the elements of the word vector.
        :param glove_dir: Path where the glove directory is located. That directory must contain
        text files with the structure mentioned above.
        :param vectors_dim: Size of the word vector. Possible values are: 50, 100, 200, 300.
        """

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
