import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from sklearn.manifold import TSNE
from wordcloud import WordCloud, STOPWORDS

from models.topics import TopicsModel


def plot_distribution_of_doc_word_counts(documents):
    """
    Plot a histogram of the number of words in the documents.
    :param documents: : List[List[str]]
    """
    docs_word_counts = [len(doc) for doc in documents]

    # Histogram
    plt.figure(figsize=(14, 8), dpi=160)
    plt.hist(docs_word_counts, bins=1000, color='navy')
    plt.text(750, 1200, "Mean   : " + str(round(np.mean(docs_word_counts))))
    plt.text(750, 1100, "Median : " + str(round(np.median(docs_word_counts))))
    plt.text(750, 1000, "Stdev   : " + str(round(np.std(docs_word_counts))))
    plt.text(750, 900, "1%ile    : " + str(round(np.quantile(docs_word_counts, q=0.01))))
    plt.text(750, 800, "99%ile  : " + str(round(np.quantile(docs_word_counts, q=0.99))))

    plt.gca().set(xlim=(0, 1000), ylabel='Number of Documents', xlabel='Document Word Count')
    plt.tick_params(size=16)
    plt.xticks(np.linspace(0, 1000, 9))
    plt.title('Distribution of Document Word Counts', fontdict=dict(size=25))
    plt.show()


def plot_word_clouds_k_keywords_each_topic(topics_model, num_topics=None, num_keywords=10):
    """
    Plots word clouds for the specified number of topics in the given model.
    :type topics_model: TopicsModel or gensim.models.wrappers.LdaMallet or gensim.models.LdaModel
    or gensim.models.LsiModel.
    :param topics_model: gensim models (lda, lsa or lda_mallet)
    :param num_topics: Number of topics to be plotted. If is None, the num_topics of the model are used.
    :param num_keywords: Number of keywords in each topic to be plotted.
    """

    # If topics_model is a TopicsModel, obtain the gensim model inside it.
    if isinstance(topics_model, TopicsModel):
        topics_model = topics_model.model

    colors = [color for color in mcolors.TABLEAU_COLORS.values()]  # List of colors
    # Index of the current topic to be plotted.
    # Is used also for selecting the color for that topic in the function below.
    topic_index = 0

    def color_func(*args, **kwargs):
        return colors[topic_index % len(colors)]

    cloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=num_keywords,
                      colormap='tab10',
                      color_func=color_func,
                      prefer_horizontal=1.0)

    if num_topics is None:
        num_topics = topics_model.num_topics

    # Obtain the topics from the model, with the specified number of topics and keywords
    topics = topics_model.show_topics(num_topics, num_keywords, formatted=False)

    # Each plot is formed by 4 subplots, each one containing the words of a topic
    for _ in range(num_topics // 4):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

        for ax in axes.flatten():
            fig.add_subplot(ax)

            topic_index = topics[topic_index][0]
            topic = topics[topic_index][1]
            topic_words = dict(topic)
            cloud.generate_from_frequencies(topic_words, max_font_size=300)

            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(topic_index), fontdict=dict(size=20))
            plt.gca().axis('off')

            topic_index += 1

        # plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.show()


def tsne_clustering_chart():
    # Get topic weights
    topic_weights = []
    for i, row_list in enumerate(lda_model[corpus]):
        topic_weights.append([w for i, w in row_list[0]])

    # Array of topic weights
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Keep the well separated points (optional)
    arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    # Plot the Topic Clusters using Bokeh
    output_notebook()
    n_topics = 4
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics),
                  plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[topic_num])
    show(plot)
