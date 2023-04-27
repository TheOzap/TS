
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime


RUN apt-get update -y && apt-get upgrade -y && apt-get install -y sudo

ARG python_version=3.8


RUN apt-get install -y python${python_version}

RUN sudo apt install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip install --upgrade pip

RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install cmake
RUN pip3 install matplotlib
RUN pip3 install pyyaml

RUN pip install -U scikit-learn
RUN pip install tensorboard --upgrade
RUN pip install yfinance
RUN pip install jupyterlab --upgrade --no-cache-dir


