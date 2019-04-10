import matplotlib.pyplot as plt
from keras.layers import Input, Conv1D, Dropout, Dense, GlobalMaxPooling1D, MaxPooling1D
from keras.models import Model

from utils import pretty_print


def print_keras_model_inputs_and_outputs(model):
    for layer in model.layers:
        print("Name: " + layer.name + ". Input shape: " + str(layer.input_shape) +
              ". Output shape: " + str(layer.output_shape))


def plot_keras_model_loss_during_epochs(history):
    # Get training and test loss histories
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def plot_keras_model_accuracy_during_epochs(history):
    # Get training and test accuracy histories
    training_acc = history.history['acc']
    test_acc = history.history['val_acc']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_acc) + 1)

    # Visualize acc history
    plt.plot(epoch_count, training_acc, 'r--')
    plt.plot(epoch_count, test_acc, 'b-')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


class CNNSentenceClassification:
    """
    This network uses the Kim topology
    1. Convolutional layer
    2. Max-over-time Pooling layer
    3. Fully conected softmax layer: it's output is the probability distribution over labels

    Other characteristics:
    - ReLU activation function
    - Filter windows of 3, 4, 5 with 100 feature maps each (num filters)
    - For regularization: Dropout on the penultimate layer with a constraint on l2-norms of the weight vectors.
                          Dropout rate of 0.50 with a L2 constraint of 3
    - Mini-batch size of 50
    - Early stopping on validation sets
    - Training with SGD over shuffled mini-batches with the Adadelta update rule
    """

    __RANDOM_STATE = 42

    __ACTIVATION_FUNCTION = 'relu'
    __NUM_FILTERS = 100
    __FILTER_WINDOW_SIZE = 5
    __DROPOUT_RATE = 0.50

    __OPTIMIZER = 'adadelta'
    __METRICS = ['acc']

    __BATCH_SIZE = 50
    __NUM_EPOCHS = 5
    __VALIDATION_SPLIT = 0.2

    def __init__(self, embedding_layer, word_vectors_max_len, num_classes, random_state=__RANDOM_STATE,
                 activation_function=__ACTIVATION_FUNCTION, num_filters=__NUM_FILTERS,
                 filter_window_size=__FILTER_WINDOW_SIZE, dropout_rate=__DROPOUT_RATE,
                 optimizer=__OPTIMIZER, metrics=__METRICS):
        sequence_input = Input(shape=(word_vectors_max_len,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(filters=num_filters, kernel_size=filter_window_size,
                   activation=activation_function, use_bias=True)(embedded_sequences)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(rate=dropout_rate, seed=random_state)(x)  # TODO: L2 regularization??
        preds = Dense(units=num_classes, activation='softmax')(x)

        self.model = Model(sequence_input, preds)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=metrics)

        self.last_train_history = None

    def train(self, x_train, y_train,
              epochs=__NUM_EPOCHS, batch_size=__BATCH_SIZE,
              validation_split=__VALIDATION_SPLIT):
        self.last_train_history = self.model.fit(x_train, y_train,
                                                 epochs=epochs, batch_size=batch_size,
                                                 validation_split=validation_split)
        return self.last_train_history

    def get_last_train_history(self):
        return self.last_train_history

    def evaluate(self, x_val, y_val, batch_size=__BATCH_SIZE):
        """
        Prints the model results on the given validation set.
        :param x_val: Validation set X
        :param y_val: Validation set y
        :param batch_size: Size of the batch.
        :returns val_loss, val_acc, if the metrics passed in the __init__ where the default.
        """
        return self.model.evaluate(x_val, y_val, batch_size=batch_size)

    # TODO: Move to a base class? El parametro random_state que está aqui pero
    #  no en Keras-Example se podria pasar como kwargs
    @classmethod
    def train_and_evaluate(cls, embedding_layer, word_vectors_max_len, dataset_num_classes,
                           random_state, X_train, y_train, X_val, y_val, epochs):
        """
        Creates and trains the CNN, plots loss and accuracy, and check the results on the validation set.
        :param embedding_layer: Embedding layer for the cnn keras model.
        :param word_vectors_max_len: Max length of the word vectors.
        :param dataset_num_classes: Number of classes of the dataset.
        :param random_state: Random state for the dropout layer.
        :param X_train: X data for training.
        :param y_train: y data for training.
        :param X_val: X data for validation.
        :param y_val: y data for training.
        :param epochs: Number of epochs
        """

        # Create and train the cnn
        pretty_print('Creating the CNN-Sentence-classification')
        model = cls(embedding_layer, word_vectors_max_len, dataset_num_classes, random_state)
        pretty_print('Training the CNN-Sentence-classification')
        history = model.train(X_train, y_train, epochs=epochs)

        # Plot loss and accuracy
        plot_keras_model_loss_during_epochs(history)
        plot_keras_model_accuracy_during_epochs(history)

        # Check results on the validation set
        pretty_print('\n\nValidation results\n____________________')
        val_loss, val_acc = model.evaluate(X_val, y_val)
        print('Validation loss:', val_loss)
        print('Validation accuracy:', val_acc)


class CNNKerasExample:  # TODO: Create a parent class??
    """
    This network follows the example explained in Keras:
    https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
    """

    __ACTIVATION_FUNCTION = 'relu'
    __NUM_FILTERS = 128
    __FILTER_WINDOW_SIZE = 5
    __POOL_SIZE = 5

    __OPTIMIZER = 'rmsprop'
    __METRICS = ['acc']

    __BATCH_SIZE = 128
    __NUM_EPOCHS = 10
    __VALIDATION_SPLIT = 0.2

    def __init__(self, embedding_layer, word_vectors_max_len, num_classes,
                 activation_function=__ACTIVATION_FUNCTION, num_filters=__NUM_FILTERS,
                 filter_window_size=__FILTER_WINDOW_SIZE, pool_size=__POOL_SIZE,
                 optimizer=__OPTIMIZER, metrics=__METRICS):
        sequence_input = Input(shape=(word_vectors_max_len,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(filters=num_filters, kernel_size=filter_window_size,
                   activation=activation_function)(embedded_sequences)
        x = MaxPooling1D(pool_size=pool_size)(x)
        x = Conv1D(filters=num_filters, kernel_size=filter_window_size, activation=activation_function)(x)
        x = MaxPooling1D(pool_size=pool_size)(x)
        x = Conv1D(filters=num_filters, kernel_size=filter_window_size, activation=activation_function)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(units=128, activation=activation_function)(x)
        preds = Dense(units=num_classes, activation='softmax')(x)

        self.model = Model(sequence_input, preds)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=metrics)

        self.last_train_history = None

    def train(self, x_train, y_train,
              epochs=__NUM_EPOCHS, batch_size=__BATCH_SIZE,
              validation_split=__VALIDATION_SPLIT):
        self.last_train_history = self.model.fit(x_train, y_train,
                                                 epochs=epochs, batch_size=batch_size,
                                                 validation_split=validation_split)

        return self.last_train_history

    def get_last_train_history(self):
        return self.last_train_history

    def evaluate(self, x_val, y_val, batch_size=__BATCH_SIZE):
        """
        Prints the model results on the given validation set.
        :param x_val: Validation set X
        :param y_val: Validation set y
        :param batch_size: Size of the batch.
        :returns val_loss, val_acc, if the metrics passed in the __init__ where the default.
        """
        return self.model.evaluate(x_val, y_val, batch_size=batch_size)

    @classmethod
    def train_and_evaluate(cls, embedding_layer, word_vectors_max_len, dataset_num_classes,
                           X_train, y_train, X_val, y_val, epochs):
        """
        Creates and trains the CNN, plots loss and accuracy, and check the results on the validation set.
        :param embedding_layer: Embedding layer for the cnn keras model.
        :param word_vectors_max_len: Max length of the word vectors.
        :param dataset_num_classes: Number of classes of the dataset.
        :param X_train: X data for training.
        :param y_train: y data for training.
        :param X_val: X data for validation.
        :param y_val: y data for training.
        :param epochs: Number of epochs
        """

        # Create and train the cnn
        pretty_print('Creating the CNN-Keras-Example')
        model = cls(embedding_layer, word_vectors_max_len, dataset_num_classes)
        pretty_print('Training the CNN-Keras-Example')
        history = model.train(X_train, y_train, epochs=epochs)

        # Plot loss and accuracy
        plot_keras_model_loss_during_epochs(history)
        plot_keras_model_accuracy_during_epochs(history)

        # Check results on the validation set
        pretty_print('Validation results')
        val_loss, val_acc = model.evaluate(X_val, y_val)
        print('Validation loss:', val_loss)
        print('Validation accuracy:', val_acc)
