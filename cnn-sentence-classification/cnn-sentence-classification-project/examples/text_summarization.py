from pprint import pprint

import networkx as nx
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from embeddings import Glove
from preprocessing.text import preprocess_text
from utils import load_obj_from_disk, pretty_print

if __name__ == '__main__':
    """
    Text summarization of single documents
    """

    # region 0. Get a document without preprocessing
    pretty_print('0. Document without preprocessing')

    # Load dataset and transform to documents list
    dataset = load_obj_from_disk('dataset')
    documents = dataset.as_documents_content_list(tokenize_words=False)

    # Get a single document
    dataset_doc_or_user_doc = input('Use a dataset doc (d) or enter your own doc (o)?')
    if dataset_doc_or_user_doc == 'd':
        doc_index = int(input('Doc index: '))
        doc = documents[doc_index]
    else:
        doc = input('Write the doc:')

    pretty_print('Document content')
    print(doc)
    # endregion

    # region 1. Split text into sentences
    pretty_print('1. Split text into sentences')
    doc_sentences = sent_tokenize(doc)
    for i, sent in enumerate(doc_sentences):
        pretty_print('Sentence {}'.format(i))
        print(sent)
    print('\n\nNum sentences: ', len(doc_sentences))
    # endregion

    # region 2. Preprocess each sentence
    pretty_print('2. Preprocess each sentence')
    preprocessed_sentences = [preprocess_text(sent) for sent in doc_sentences]
    for i, sent in enumerate(preprocessed_sentences):
        pretty_print('Sentence {}'.format(i))
        print(sent)
    print('\n\nNum sentences: ', len(preprocessed_sentences))
    # endregion

    # region 3. Transform each sentence into a word embedding vector
    pretty_print('3. Transform each sentence into a word embedding vector')

    # region 3.1. GloVe
    pretty_print('3.1. GloVe')

    EMBEDDINGS_DIM = 100  # Size of the word embeddings vectors
    glove = Glove(EMBEDDINGS_DIM)
    print('Found %s word vectors.' % len(glove.embeddings))

    # Transform each word of each sentence into a vector and calculate the mean
    # of the vectors of the sentence. That mean vector will be the sentence vector.
    sentence_vectors = []
    for sent in preprocessed_sentences:
        sent_words = sent.split()
        # If the sentence has no words after the preprocessing a default numpy array full of zeros is used
        if len(sent_words) == 0:
            sent_vector = np.zeros(EMBEDDINGS_DIM)
        else:
            # For each word in the sentence, transform it to a vector. If the word is not present
            # in the glove embeddings, a default numpy array full of zeros is used.
            word_vectors = [glove.get_word_vector(word) for word in sent_words]
            # Sentence vector is the mean of the word vectors
            sent_vector = sum(word_vectors) / len(word_vectors)

        sentence_vectors.append(sent_vector)

    pretty_print('Sentence vectors')
    for i, vector in enumerate(sentence_vectors):
        pretty_print('Sentence vector {}'.format(i))
        print(vector)
    # endregion

    # region 3.2. Word2Vec
    # TODO
    # endregion

    # endregion

    # region 4. Find similarities between sentences (Similarity Matrix)
    pretty_print('4. Find similarities between sentences (Similarity Matrix)')

    # TODO: Create a triangular matrix? Saves space and time

    # Define a matrix full of 0's of size MxM, being M the number of sentences
    similarity_matrix = np.zeros([len(sentence_vectors), len(sentence_vectors)])

    # Fill the matrix with the values of the cosine similarity between sentences
    for i in range(len(sentence_vectors)):
        for j in range(len(sentence_vectors)):
            # If the sentence in the row and in the column is distinct
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(
                    sentence_vectors[i].reshape(1, 100),
                    sentence_vectors[j].reshape(1, 100)
                )[0, 0]

    # endregion

    # region 5. Transform the similarity matrix into a graph
    pretty_print('5. Transform the similarity matrix into a graph')
    nx_graph = nx.from_numpy_array(similarity_matrix)

    # endregion

    # region 6. Apply TextRank to the graph
    pretty_print('6. Apply TextRank to the graph')
    sentence_scores = nx.pagerank(nx_graph)
    pprint(sentence_scores)
    # endregion

    # region 7. Order sentences by the Page Rank score
    pretty_print('7. Order sentences by the Page Rank score')
    sorted_sentences = sorted(((sentence_scores[i], sent) for i, sent in enumerate(doc_sentences)), reverse=True)
    pprint(sorted_sentences)
    # endregion

    # region 8. Get k best sentences
    pretty_print('8. Get k best sentences')
    k = int(input('Num best sentences: '))
    for i in range(k):
        print()
        print(' '.join(sorted_sentences[i][1].split()))
    # endregion
