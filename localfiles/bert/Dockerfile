#Below commands are require while running locally
FROM continuumio/anaconda3:latest
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch
RUN pip install transformers
RUN pip install tensorflow==2.5.1
COPY . /ml/
WORKDIR /ml
CMD  python train.py




#Below commands are required while running on the sagemaker.
#FROM tensorflow/tensorflow:2.5.0-gpu-jupyter
#RUN pip install pandas
#RUN pip install torchvision
#RUN pip install scikit-learn
#RUN pip3 install sagemaker-training

#COPY . /opt/ml/code/
#ENV SAGEMAKER_PROGRAM train.py




