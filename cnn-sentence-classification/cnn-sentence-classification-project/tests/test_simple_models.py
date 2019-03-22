from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import preprocessing as pre
from datasets.twenty_news_groups import TwentyNewsGroupsDataset
from embeddings import Word2VecModel
from evaluation import evaluate_model
from utils import pretty_print

if __name__ == '__main__':
    dataset = TwentyNewsGroupsDataset()
    dataset.print_some_files()

    # Apply more preprocessing
    pretty_print('Remove stopwords')
    dataset.apply_function_to_files(pre.to_lowercase_and_remove_stopwords)
    dataset.print_some_files()

    pretty_print('Remove punctuation')
    dataset.apply_function_to_files(pre.substitute_punctuation)
    dataset.print_some_files()

    pretty_print('Lemmatization')
    dataset.apply_function_to_files(pre.lemmatize_words)
    dataset.print_some_files()

    pretty_print('As Pandas Dataframe')
    df = dataset.as_dataframe()
    print(df.describe())
    print(df.head(5))

    pretty_print('Word2Vec test')
    w2vmodel = Word2VecModel()
    print('Hello as word2vec vector: ', w2vmodel.get_word_vector('Hello'))

    pretty_print('Split into train and validation sets')
    RANDOM_STATE = 42
    X_train, X_val, y_train, y_val = train_test_split(
        df['document'], df['class'], test_size=0.2, random_state=RANDOM_STATE)

    pretty_print('Transform the documents from text to vectors using TF IDF')
    tfidf_vectorizer = TfidfVectorizer()
    # Learn vocabulary and idf from training set and return term-document matrix
    vectors_train = tfidf_vectorizer.fit_transform(X_train)
    # Transform documents to document-term matrix.
    vectors_val = tfidf_vectorizer.transform(X_val)

    # TODO: Hyper-parameter tunning
    pretty_print('Train a Naive Bayes model')
    nb_clf = MultinomialNB(alpha=0.01)
    nb_clf.fit(vectors_train, y_train)

    pretty_print('Evaluate Naive Bayes performance on the validation set')
    evaluate_model(nb_clf, vectors_val, y_val, dataset.classes)

    pretty_print('Train a Linear SVM model')
    svm_clf = LinearSVC(random_state=42)
    svm_clf.fit(vectors_train, y_train)

    pretty_print('Evaluate Linear SVM performance on the validation set')
    evaluate_model(svm_clf, vectors_val, y_val, dataset.classes)
