#  $ docker build . -t topics_and_summary:latest
#  $ docker run --name topics_and_summary -v $PWD/demo-images:/topics_and_summary/demo-images -i -t topics_and_summary:latest

FROM ubuntu:latest
VOLUME /topics_and_summary/demo-images


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# add /topics_and_summary to the PYTHONPATH. This is needed to import topics_and_summary without installing it.
ENV PYTHONPATH=/topics_and_summary

RUN apt-get update --fix-missing && \
    # Install python3.6 and pip3
    apt-get install -y --no-install-recommends python3.6 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy all the files of the project (except the ones specified in the .dockerignore)
# into the /topics_and_summary folder of the docker image
COPY $PWD /topics_and_summary

# Install the python packages needed for generating the binary distributions
RUN pip3 install wheel setuptools

# Create the binary distribution of topics_and_summary
RUN cd /topics_and_summary && \
    python3 setup.py bdist_wheel && \
    # Install the wheel file with the topics_and_summary library to install the required packages.
    pip3 install dist/*.whl

# Install nltk resources
RUN python3 -m nltk.downloader stopwords wordnet punkt

# Install java
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install default-jdk -y

# Run the demo at the start of the container, specifying as first argument the path to the demo-docker-conf.ini file
CMD python3 /topics_and_summary/topics_and_summary/examples/demo.py /topics_and_summary/demo-docker-conf.ini
