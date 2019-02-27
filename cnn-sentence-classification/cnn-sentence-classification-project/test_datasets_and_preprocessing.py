from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from datasets import TwentyNewsGroupsDataset
import preprocessing as pre
from embeddings import Word2VecModel


if __name__ == '__main__':
    dataset = TwentyNewsGroupsDataset()
    print(dataset.print_some_files())

    # Apply more preprocessing
    print('\n\nRemove stopwords\n________________')
    dataset.apply_function_to_files(pre.to_lowercase_and_remove_stopwords)
    print(dataset.print_some_files())

    print('\n\nRemove punctuation\n________________')
    dataset.apply_function_to_files(pre.substitute_punctuation)
    print(dataset.print_some_files())

    print('\n\nLemmatization\n________________')
    dataset.apply_function_to_files(pre.lemmatize_words)
    print(dataset.print_some_files())

    print('\n\nAs Pandas Dataframe\n________________')
    df = dataset.as_dataframe()
    print(df.describe())
    print(df.head(5))

    print('\n\nWord2Vec test\n________________')
    w2vmodel = Word2VecModel()
    print('Hello as word2vec vector: ', w2vmodel.get_word_vector('Hello'))
