from sklearn import metrics
from sklearn.metrics import classification_report


def evaluate_model(model, vectors_test, y_test, class_labels):
    """
    Prints the F1 score and the classification report on the training set of the passed model.
    """
    predictions = model.predict(vectors_test)
    print('F1 score:', metrics.f1_score(y_test, predictions, average='macro'))
    print()
    print('Classification report\n_____________________')
    print(classification_report(y_test, predictions, target_names=class_labels))
