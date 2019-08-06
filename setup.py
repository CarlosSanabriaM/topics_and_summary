from os import path

from setuptools import setup, find_packages

this_directory_abs_path = path.abspath(path.dirname(__file__))
with open(path.join(this_directory_abs_path, 'README.md'), "r") as fh:
    long_description = fh.read()

setup(name='topics_and_summary',
      version='1.0',
      description='Package for identifying the topics present in a collection '
                  'of text documents and create summaries of texts',
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords="topics summarization text NLP lda lsa textrank",

      url='https://github.com/CarlosSanabriaM/topics_and_summary',
      project_urls={
          "Source Code": "https://github.com/CarlosSanabriaM/topics_and_summary",
          # TODO: Update when the doc is in ReadTheDocs
          # "Documentation": "<URL>"
      },

      author='Carlos Sanabria Miranda',
      author_email='uo250707@uniovi.es',

      install_requires=[
          'numpy==1.15.4',
          'gensim==3.4.0',
          'matplotlib==3.0.2',
          'seaborn==0.9.0',
          'tqdm==4.31.1',
          'texttable==1.6.1',
          'dill==0.2.9',
          'nltk==3.3',
          'bokeh==1.0.4',
          'networkx==2.2',
          'wordcloud==1.5.0',
          'pandas==0.24.1',
          'scikit_learn==0.20.3',
          'typing==3.6.6'
      ],
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
