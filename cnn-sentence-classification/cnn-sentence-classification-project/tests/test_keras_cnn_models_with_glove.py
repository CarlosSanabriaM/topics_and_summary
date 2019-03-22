# %%
import numpy as np
from keras.initializers import Constant
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from datasets.twenty_news_groups import TwentyNewsGroupsDataset
from embeddings import Glove
from models.classification import CNNSentenceClassification, CNNKerasExample
from preprocessing.dataset import preprocess_dataset
from utils import pretty_print


def encode_class(laber_encoder):
    """
    Creates a new column with the class encoded as an integer from 0 to num_classes-1.
    :return: that column encoded using one-hot-encoding.
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

    # %%
    # Transform each document into a list of word indices

    # Maximum number of words to keep, based on word frequency.
    # Only the most common MAX_NUM_WORDS-1 words will be kept.
    MAX_NUM_WORDS = 20000

    texts = df['document'].to_list()
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Store in word_index a dict, where keys are words an values it's corresponding index
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # 0 is a reserved index that won't be assigned to any word.

    # %%
    # Limit (and pad if needed) the size of each document to 1000 words
    WORD_VECTORS_MAX_LEN = 1000
    padded_words_indices_list = pad_sequences(sequences, maxlen=WORD_VECTORS_MAX_LEN, padding='post')

    # Obtain the class column as a numpy matrix, where each row is the class string in the one-hot-encoding
    laber_encoder = LabelEncoder()
    class_one_hot_encoded = encode_class(laber_encoder)
    print('Shape of padded_words_indices_list tensor:', padded_words_indices_list.shape)
    print('Shape of class_one_hot_encoded tensor:', class_one_hot_encoded.shape)

    # %%
    # Separate into train and validation sets
    RANDOM_STATE = 42
    X_train, X_val, y_train, y_val = train_test_split(
        padded_words_indices_list, class_one_hot_encoded, test_size=0.2, random_state=RANDOM_STATE)

    print('Shape of X_train:', X_train.shape)
    print('Shape of X_val:', X_val.shape)
    print('Shape of y_train:', y_train.shape)
    print('Shape of y_val:', y_val.shape)

    # %%
    # Create the Glove model
    EMBEDDING_DIM = 100
    glove = Glove(EMBEDDING_DIM)
    print('Found %s word vectors.' % len(glove.embeddings_index))

    # %%
    # Prepare the embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1  # TODO: Quitar el +1 ??
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        # TODO: Se puede optimizar poniendo break?? Si word_index.items() está ordenado sí.
        if i > MAX_NUM_WORDS:
            continue
        embedding_vector = glove.embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print('Shape of embedding_matrix:', embedding_matrix.shape)

    # %%
    # Create the embedding layer
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=WORD_VECTORS_MAX_LEN,
                                trainable=False)

    # %%
    CNNSentenceClassification.train_and_evaluate(embedding_layer, WORD_VECTORS_MAX_LEN, dataset.num_classes,
                                                 RANDOM_STATE, X_train, y_train, X_val, y_val, epochs=20)

    # %%
    CNNKerasExample.train_and_evaluate(embedding_layer, WORD_VECTORS_MAX_LEN, dataset.num_classes,
                                       X_train, y_train, X_val, y_val, epochs=20)
