import abc
from typing import List

import networkx as nx
import numpy as np
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from embeddings import Glove, Word2VecModel
from preprocessing.text import preprocess_text


class SummarizationModel(metaclass=abc.ABCMeta):
    """Interface that represents a summarization model."""

    @abc.abstractmethod
    def get_k_best_sentences_of_text(self, text: str, num_best_sentences=5) -> List[str]:
        """
        Get the k best sentences of the given text.
        :param text: Text where summary sentences will be obtained.
        :param num_best_sentences: Number of summary sentences to be returned.
        :return: List[str] where each str is a sentence.
        """


class TextRank(SummarizationModel):
    """Summarization class using the TextRank algorithm"""

    MAX_NUM_ITERATIONS = 500
    """
    Max number of iterations in the power method eigenvalue solver. If the algorithm fails to converge to the
    specified tolerance within the specified number of iterations of the power iteration method, the
    PowerIterationFailedConvergence is raised.
    """

    def __init__(self, embedding_model='glove', glove_embedding_dim=100):
        """
        Creates a TextRank object that uses the specified embedding model to create the sentence vectors.
        :param embedding_model: Embedding model used to create the sentence vectors.
        Possible values are: 'glove' or 'word2vec'.
        :param glove_embedding_dim: Dimension of the embedding word vectors.
        """
        if embedding_model == 'glove':
            self.embedding_model = Glove(glove_embedding_dim)
        else:
            self.embedding_model = Word2VecModel()

    # TODO: Sometimes it can't converge, so it throws an Exception
    def get_k_best_sentences_of_text(self, text: str, num_best_sentences=5) -> List[str]:
        """
        Get the k best sentences of the given text. This method performs an Extractive Summary of the given text.
        :param text: Text where sentences will be extracted.
        :param num_best_sentences: Number of sentences of the text to be returned.
        :return: List[str] where each str is a sentence.
        The list is in descending order by the importance of the sentences.
        It can raise a PowerIterationFailedConvergence exception if the algorithm doesn't converge.
        """

        # 1. Split text into sentences
        text_sentences = sent_tokenize(text)

        # 2. Preprocess each sentence
        preprocessed_sentences = [preprocess_text(sent) for sent in text_sentences]

        # 3. Transform each sentence into a word embedding vector
        # Transform each word of each sentence into a vector and calculate the mean
        # of the vectors of the sentence. That mean vector will be the sentence vector.
        sentence_vectors = []
        for sent in preprocessed_sentences:
            sent_words = sent.split()
            # If the sentence has no words after the preprocessing a default numpy array full of zeros is used
            if len(sent_words) == 0:
                sent_vector = np.zeros(self.embedding_model.get_vectors_dim())
            else:
                # For each word in the sentence, transform it to a vector. If the word is not present
                # in the embeddings model, a default numpy array full of zeros is used.
                word_vectors = [self.embedding_model.get_word_vector(word) for word in sent_words]
                # Sentence vector is the mean of the word vectors
                sent_vector = sum(word_vectors) / len(word_vectors)

            sentence_vectors.append(sent_vector)

        # 4. Find similarities between sentences (Similarity Matrix)
        # Define a matrix full of 0's of size MxM, being M the number of sentences
        similarity_matrix = np.zeros([len(sentence_vectors), len(sentence_vectors)])

        # Fill the matrix with the values of the cosine similarity between sentences
        for i in range(len(sentence_vectors)):
            for j in range(len(sentence_vectors)):
                # If the sentence in the row and in the column is distinct
                if i != j:
                    similarity_matrix[i][j] = cosine_similarity(
                        sentence_vectors[i].reshape(1, self.embedding_model.get_vectors_dim()),
                        sentence_vectors[j].reshape(1, self.embedding_model.get_vectors_dim())
                    )[0, 0]

        # 5. Transform the similarity matrix into a graph
        nx_graph = nx.from_numpy_array(similarity_matrix)

        # 6. Apply TextRank to the graph
        sentence_scores = nx.pagerank(nx_graph, max_iter=self.MAX_NUM_ITERATIONS)

        # 7. Order sentences by the Page Rank score
        sorted_sentences = sorted(((sentence_scores[i], sent) for i, sent in enumerate(text_sentences)), reverse=True)

        # 8. Get k best sentences
        return list(map(lambda sent_tuple: ' '.join(sent_tuple[1].split()), sorted_sentences[:num_best_sentences]))
