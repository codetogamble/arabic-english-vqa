# Use an official PyTorch runtime as a parent image
# FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04


RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get --assume-yes --no-install-recommends install \
        build-essential \
        curl \
        git \
        jq \
        libgomp1 \
        zlib1g-dev \
        libssl-dev \
        libffi-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libreadline-dev \
        libbz2-dev \
        liblzma-dev \
        libsqlite3-dev \
        wget \
        vim

RUN apt update



ENV PYTHON_VERSION=3.8.17
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
RUN tar -xf Python-${PYTHON_VERSION}.tgz

WORKDIR Python-${PYTHON_VERSION}

RUN ./configure
RUN make altinstall

RUN rm -rf Python-${PYTHON_VERSION}

RUN apt-get autoremove


RUN pip3.8 install --no-cache-dir --upgrade pip==22.3.1
RUN ln -s /usr/local/bin/python3.8 /usr/bin/python & \
    ln -s /usr/local/bin/pip3.8 /usr/bin/pip

WORKDIR /usr/src/app


COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /usr/src/app

RUN mkdir /usr/src/data

ENV DATA_DIR=/usr/src/data
ENV MODEL_PATH=/usr/src/app/default_model

RUN pyarmor-7 obfuscate finetune.py server_lic.py
RUN rm *.py
RUN rm Dockerfile
RUN apt update && apt install poppler-utils -y

EXPOSE 80
