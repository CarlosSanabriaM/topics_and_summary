import matplotlib.pyplot as plt
import seaborn as sns

from datasets.twenty_news_groups import TwentyNewsGroupsDataset
from preprocessing.dataset import preprocess_dataset
from utils import pretty_print

if __name__ == '__main__':
    dataset = TwentyNewsGroupsDataset()
    dataset, trigrams_func = preprocess_dataset(dataset, ngrams='tri')
    df = dataset.as_dataframe()

    # Create a new column with a list of the words in each document
    df['num_words'] = df['document'].apply(lambda x: len(x.split()))

    # Obtain statistics on the number of words in each document
    pretty_print('Stats on the number of words in each document')
    print(df['num_words'].describe())

    # Print percentiles
    print()
    print('80th percentile: ', df['num_words'].quantile(0.80))
    print('85th percentile: ', df['num_words'].quantile(0.85))
    print('90th percentile: ', df['num_words'].quantile(0.90))
    print('95th percentile: ', df['num_words'].quantile(0.95))

    # Plot a boxplot of the num words in each document
    sns.set(style="whitegrid")
    sns.boxplot(x=df['num_words'])
    plt.show()
