from setuptools import setup

setup(name='topics_and_summary',
      version='0.1',
      description='Package for identifying the topics present in a collection '
                  'of text documents and create summaries of texts',
      keywords="topics summarization text NLP lda lsa textrank",

      url='madbd1000.accenture.com:8081/carlos.sanabria/topic_and_extractivesummary',
      project_urls={
          "Source Code": "madbd1000.accenture.com:8081/carlos.sanabria/topic_and_extractivesummary",
          "Documentation": "madbd1000.accenture.com:8081/carlos.sanabria/topic_and_extractivesummary/docs"
      },

      author='Carlos Sanabria Miranda',
      author_email='carlos.sanabria@accenture.com',

      install_requires=[
            'numpy>=1.15.4',
            'gensim>=3.4.0',
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
