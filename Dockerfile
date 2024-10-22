# FROM --platform=linux/amd64 ubuntu:20.04 AS builder
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y openjdk-8-jdk wget software-properties-common

# If M-chip Mac, use arm-64; Intel-Chip => amd64
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-arm64
# ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

ENV PATH=$PATH:$JAVA_HOME/bin

RUN apt-get install -y python3 python3-pip

# 添加 pkg-config 的安装
RUN apt-get update && \
    apt-get install -y pkg-config

RUN wget https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz && \
    tar xvf spark-3.1.2-bin-hadoop3.2.tgz && \
    mv spark-3.1.2-bin-hadoop3.2 /spark && \
    rm spark-3.1.2-bin-hadoop3.2.tgz

RUN apt-get install -y gnupg2 && \
    wget www.scala-lang.org/files/archive/scala-2.12.10.deb && \
    dpkg -i scala-2.12.10.deb && \
    rm scala-2.12.10.deb

ENV SPARK_HOME=/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

RUN pip3 install pyspark==3.1.2
RUN pip3 install pandas

# RUN pip3 install xgboost==0.72.1 # Vocareum use 0.7.2.1
RUN pip3 install xgboost==2.0.1
RUN pip install numpy
RUN pip install scikit-learn

## PROJECT
RUN pip install node2vec
RUN pip install networkx
RUN pip install tqdm
RUN pip install lightgbm
RUN pip install catboost
RUN pip install matplotlib
RUN pip install gensim

# 更新系统包并安装 HDF5 库
RUN apt-get update && \
    apt-get install -y pkg-config libhdf5-dev
# RUN pip install tensorflow
# RUN pip install keras

## SCRIPT
# RUN pip install cryptography
RUN pip install nodejs npm
RUN pip3 install --no-cache-dir jupyter
RUN pip install seaborn
RUN pip install imblearn
RUN pip install community
RUN pip install python-louvain
RUN pip install spacy
RUN pip install hyperopt
EXPOSE 8889
WORKDIR /workspace

# CMD tail -f /dev/null
CMD ["/bin/bash"]