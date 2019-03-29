import math

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
from utils import RANDOM_STATE, now_as_str, join_paths, get_abspath_from_project_root


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


# TODO: Add a param that receives a dataframe, to avoid recalculating the matrix if dataframe is available
def plot_word_clouds_k_keywords_each_topic(topics_model, num_topics=None, num_keywords=10,
                                           save=False, dir_save_path=None, dpi=350, show_plot=True):
    """
    Plots word clouds for the specified number of topics in the given model.
    :type topics_model: TopicsModel or gensim.models.wrappers.LdaMallet or gensim.models.LdaModel
    or gensim.models.LsiModel.
    :param topics_model: gensim models (lda, lsa or lda_mallet)
    :param num_topics: Number of topics to be plotted. If is None, the num_topics of the model are used.
    :param num_keywords: Number of keywords in each topic to be plotted.
    :param save: If true, the plots are saved to disk.
    :param dir_save_path: If save is True, this is the path of the directory where the plots will be saved.
    :param dpi: Dots per inches for the images.
    :param show_plot: If true, shows the plot while executing.
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
    for i in range(math.ceil(num_topics / 4)):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=dpi, sharex=True, sharey=True)

        for ax in axes.flatten():
            # If the topic index equals the num_topics, the current plot has less than 4 topic to show,
            # so we remove that axes from the plot.
            if topic_index == num_topics:
                fig.delaxes(ax)
                continue

            fig.add_subplot(ax)

            topic_index = topics[topic_index][0]
            topic = topics[topic_index][1]
            topic_words = dict(topic)
            cloud.generate_from_frequencies(topic_words,
                                            max_font_size=300)  # TODO: Process finished with exit code 139 (interrupted by signal 11: SIGSEGV) when LSAModel is used

            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(topic_index), fontdict=dict(size=20))
            plt.gca().axis('off')

            topic_index += 1

        # plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()

        if save:
            plot_path = join_paths(dir_save_path, 'wordcloud{}.png'.format(i))
            plt.savefig(plot_path, dpi=dpi)

        if show_plot:
            plt.show()

        plt.clf()  # TODO: Does this avoid showing the plots when show_plot is False and plt.show() is called in another part?


__TSNE_SAVE_PATH = get_abspath_from_project_root('saved-models/topics/tsne')


def tsne_clustering_chart(model: TopicsModel, num_dimensions=2, angle=.99, doc_threshold=0,
                          plot_keywords=True, num_keywords=5, keywords_color_is_black=True,
                          save_path=__TSNE_SAVE_PATH, plot_name=None, show_plot=True):
    """
    Use t-SNE technique for dimensionality reduction.
    :param model: Topics Model.
    :param num_dimensions: Number of dimensions of the tSNE result. Should be 2 or 3.
    :param angle: Number between 0 and 1. Angle less than 0.2 has quickly increasing computation
    time and angle greater 0.8 has quickly increasing error.
    :param doc_threshold: Threshold that each document has to pass to be added to the plot.
    :param plot_keywords: If True, the keywords of each topic are plotted near a document of the topic.
    :param num_keywords: Number of keyword to show if plot_keywords is True.
    :param keywords_color_is_black: If true, the keywords color is black. If not, is the same color as the topic.
    :param save_path: Path where the html file with the interactive plot will be saved.
    :param plot_name: Name of the plot to be saved.
    :param show_plot: If true, opens a browser and shows the html with the plot.
    """

    # TODO: 3d?

    # Get doc topic prob matrix
    doc_topic_prob_matrix = model.get_doc_topic_prob_matrix()

    # Don't use docs that don't pass the threshold
    _idx = np.amax(doc_topic_prob_matrix, axis=1) > doc_threshold  # idx of doc that above the threshold
    doc_topic_prob_matrix = doc_topic_prob_matrix[_idx]

    # tSNE Dimension Reduction: 20-D -> 2-D or 3-D
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

    # Configure the default output state to generate output saved to a file when show() is called.
    if plot_name is None:
        now = now_as_str()
        plot_name = 'tsne_' + now + '.html'

    bp.output_file(join_paths(save_path, plot_name), mode='inline')

    # Create the plot for the Topic Clusters using Bokeh
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(model.num_topics),
                  tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",  # plot option tools
                  plot_width=1400, plot_height=900)

    plot.scatter(x='x', y='y', color='color',
                 # When source is provided, the kwargs above must refer to keys in the dict passed to source
                 source=bp.ColumnDataSource({
                     "x": tsne_lda[:, 0],
                     "y": tsne_lda[:, 1],
                     "topic index": dominant_topic_per_doc,
                     "topic prob": dominant_topic_prob_per_doc,
                     "doc text": list(map(lambda x: ' '.join(x), model.documents[:doc_topic_prob_matrix.shape[0]])),
                     "color": colormap[dominant_topic_per_doc]
                 }))

    if plot_keywords:
        # Plot the keywords for each topic:

        # Randomly choose a doc (within a topic) coordinate as the keywords coordinate
        topic_coord = np.empty((doc_topic_prob_matrix.shape[1], num_dimensions)) * np.nan
        for topic_num in dominant_topic_per_doc:
            if not np.isnan(topic_coord).any():
                break
            topic_coord[topic_num] = tsne_lda[dominant_topic_per_doc.index(topic_num)]

        # List of num_topics keywords as a str per each topic in the model
        topics_kws = [model.get_k_kws_per_topic_as_str(topic, num_keywords) for topic in range(model.num_topics)]

        # Plot the keywords
        for i in range(doc_topic_prob_matrix.shape[1]):
            if keywords_color_is_black:
                text_color = ['#000000']
            else:
                # TODO: The library doesn't permit put a color in the contour,
                #  so this option doesn't let to visualize the words correctly
                text_color = [colormap[i]]

            plot.text(x='x', y='y', text='text', text_color='text_color',
                      source=bp.ColumnDataSource({
                          "x": [topic_coord[i, 0]],
                          "y": [topic_coord[i, 1]],
                          "text": [topics_kws[i]],
                          "topic index": [i],
                          "text_color": text_color
                      }))

    # Add info box for each doc using hover tools
    hover = plot.select(dict(type=HoverTool))
    # With @ we refer to keys in the source dict. If the key contains spaces, it must be specified like @{key name}
    # TODO: This shows this fields for all objects, including the text, that doesn't have all them, but I think
    #  there is no solution to this, or at least in the documentation they only explain how to apply tooltips to figure.
    hover.tooltips = [
        ("doc_index", "$index"),
        ("topic_index", "@{topic index}"),
        ("topic_prob", "@{topic prob}"),
        ("doc_text", "@{doc text}")
    ]

    if show_plot:
        show(plot)

    bp.save(plot)
