# %%
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from datasets.twenty_news_groups import TwentyNewsGroupsDataset
from embeddings import Word2VecModel
from models.classification import CNNSentenceClassification, CNNKerasExample
from preprocessing.dataset import preprocess_dataset
from preprocessing.text import get_first_k_words
from utils import pretty_print


def limit_documents_size(max_len):
    """
    Limits the number of words of each document,
    and creates a column in the dataframe with the number of words of each document.
    :param max_len: Max number of words of each document.
    """
    df['document'] = df['document'].apply(lambda x: get_first_k_words(x, max_len))
    # Create a column with the number of words of each document
    df['num_words'] = df['document'].apply(lambda x: len(x.split()))


def get_padded_words_indices_list(max_len):
    """
    Creates a column with a list of the word2vec indeces of each word in the document,
    transform the words_indices column to a list of lists,
    applies a padding to each list of words_indices,
    and return that final list.
    :return: A padded
    """
    # Create a new column with a list of the word2vec indeces of each word in the document
    df['words_indices'] = df['document'].apply(lambda doc:
                                               [w2vmodel.get_word_index(word) for word in doc.split()])
    # Transform the words_indices column to a list of lists
    words_indices_list = df['words_indices'].to_numpy().tolist()  # List of lists of the words indices
    # Apply a padding to each list of words_indices. This way, all the words_indices lists have the same length
    return pad_sequences(words_indices_list, maxlen=max_len, padding='post')


def encode_class(laber_encoder):
    """
    Creates a new column with the class encoded as an integer from 0 to num_classes-1.
    After that, it returns that column encoded using one-hot-encoding.
    :return:
    """
    # Encode the class from string to a int from 0 to 19
    laber_encoder.fit_transform(df['class'])
    df['class_encoded'] = laber_encoder.fit_transform(df['class'])
    # Transform the encoded class as a number to a one-hot-encoding
    return to_categorical(df['class_encoded'])


def get_class_as_string_from_one_hot_encoding(laber_encoder, one_hot_vector):
    laber_encoder.inverse_transform(one_hot_vector)


# %%


if __name__ == '__main__':
    # %%
    dataset = TwentyNewsGroupsDataset()

    # Apply more preprocessing
    dataset = preprocess_dataset(dataset)

    # Convert to a Pandas dataframe
    pretty_print('Converting dataset to Pandas Dataframe')
    df = dataset.as_dataframe()

    # Limit the size of each document to 200 words
    WORD_VECTORS_MAX_LEN = 200
    limit_documents_size(WORD_VECTORS_MAX_LEN)  # TODO: Probar con mas tamaño o a coger los de mayor tfidf

    # %%
    # Create the word2vec model
    pretty_print('Creating the Word2Vec model')
    w2vmodel = Word2VecModel()

    # Obtain from the data the list of word indices of each document, padded if needed
    padded_words_indices_list = get_padded_words_indices_list(WORD_VECTORS_MAX_LEN)

    # Obtain the class column as a numpy matrix, where each row is the class string in the one-hot-encoding
    laber_encoder = LabelEncoder()
    class_one_hot_encoded = encode_class(laber_encoder)

    # Separate into train and validation sets
    RANDOM_STATE = 42
    X_train, X_val, y_train, y_val = train_test_split(
        padded_words_indices_list, class_one_hot_encoded, test_size=0.2, random_state=RANDOM_STATE)

    # %%
    # Create the embedding layer
    embedding_layer = w2vmodel.get_keras_embedding(train_embeddings=False)

    # %%
    CNNSentenceClassification.train_and_evaluate(embedding_layer, WORD_VECTORS_MAX_LEN, dataset.num_classes,
                                                 RANDOM_STATE, X_train, y_train, X_val, y_val, epochs=20)

    # %%
    CNNKerasExample.train_and_evaluate(embedding_layer, WORD_VECTORS_MAX_LEN, dataset.num_classes,
                                       X_train, y_train, X_val, y_val, epochs=20)
