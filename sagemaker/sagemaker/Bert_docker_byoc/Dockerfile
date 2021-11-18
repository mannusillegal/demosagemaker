FROM tensorflow/tensorflow:2.5.0-gpu-jupyter
#FROM continuumio/anaconda3:latest
RUN pip install pandas
#RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch
RUN pip install torchvision
RUN pip install transformers
RUN pip install scikit-learn
RUN pip3 install sagemaker-training

COPY . /opt/ml/code/

ENV SAGEMAKER_PROGRAM train.py


