import bokeh.plotting as bp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from bokeh.models import HoverTool
from bokeh.plotting import figure, show
from sklearn.manifold import TSNE
from tqdm import tqdm
from wordcloud import WordCloud, STOPWORDS

from models.topics import TopicsModel
from utils import RANDOM_STATE, now_as_str, get_abspath


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


def tsne_clustering_chart(model: TopicsModel, num_dimensions=2, angle=.99, doc_threshold=0,
                          plot_keywords=True, num_keywords=5, plot_name=None):
    """
    Use t-SNE technique for dimensionality reduction.
    :param model:
    :param num_dimensions:
    :param angle:
    :param doc_threshold:
    :param plot_keywords:
    :param num_keywords:
    :param plot_name:
    :return:
    """
    # Get doc topic prob matrix
    doc_topic_prob_matrix = model.get_doc_topic_prob_matrix()

    # Don't use docs that don't pass the threshold
    _idx = np.amax(doc_topic_prob_matrix, axis=1) > doc_threshold  # idx of doc that above the threshold
    doc_topic_prob_matrix = doc_topic_prob_matrix[_idx]

    # tSNE Dimension Reduction: 20-D -> 2-D
    tsne_model = TSNE(n_components=num_dimensions, verbose=1, random_state=RANDOM_STATE, angle=angle, init='pca')
    tsne_lda = tsne_model.fit_transform(doc_topic_prob_matrix)

    # Colors for the points in the Bokeh plot
    colormap = np.array([
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
    ])

    # Get the most relevant topic for each doc
    dominant_topic_per_doc = []
    dominant_topic_prob_per_doc = []
    for dominant_topic_doc in tqdm(doc_topic_prob_matrix):
        dominant_topic_per_doc.append(dominant_topic_doc.argmax())
        dominant_topic_prob_per_doc.append(dominant_topic_doc.max())

    # Create the plot for the Topic Clusters using Bokeh
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(model.num_topics),
                  tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",  # plot option tools
                  plot_width=1300, plot_height=1000)

    plot.scatter(x='x', y='y', color="color",
                 # When source is provided, the kwargs above must refer to keys in the dict passed to source
                 source=bp.ColumnDataSource({
                     "x": tsne_lda[:, 0],
                     "y": tsne_lda[:, 1],
                     "topic index": dominant_topic_per_doc,
                     "topic prob": dominant_topic_prob_per_doc,
                     "doc text": list(map(lambda x: ' '.join(x), model.documents)),
                     "color": colormap[dominant_topic_per_doc]
                 }))

    if plot_keywords:
        # Plot the keywords for each topic:

        # Randomly choose a doc (within a topic) coordinate as the keywords coordinate
        topic_coord = np.empty((doc_topic_prob_matrix.shape[1], 2)) * np.nan
        for topic_num in dominant_topic_per_doc:
            if not np.isnan(topic_coord).any():
                break
            topic_coord[topic_num] = tsne_lda[dominant_topic_per_doc.index(topic_num)]

        # List of num_topics keywords as a str per each topic in the model
        topics_kws = [model.get_k_kws_per_topic_as_str(topic, num_keywords) for topic in range(model.num_topics)]

        # Plot the keywords
        for i in range(doc_topic_prob_matrix.shape[1]):
            plot.text(topic_coord[i, 0], topic_coord[i, 1], [topics_kws[i]])

    # Add info box for each doc using hover tools
    hover = plot.select(dict(type=HoverTool))
    # With @ we refer to keys in the source dict. If the key contains spaces, it must be specified like @{key name}
    hover.tooltips = [
        ("doc_index", "$index"),
        ("topic_index", "@{topic index}"),
        ("topic_prob", "@{topic prob}"),
        ("doc_text", "@{doc text}")
    ]

    # Show the plot
    show(plot)

    # Save the plot
    __TSNE_SAVE_PATH = 'saved-model/topics/tsne/'
    if plot_name is None:
        now = now_as_str()
        plot_name = 'tsne_' + now

    plot_name += '.html'
    bp.save(plot, get_abspath(__file__, __TSNE_SAVE_PATH + plot_name))
