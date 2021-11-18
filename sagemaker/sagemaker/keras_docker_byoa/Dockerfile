FROM tensorflow/tensorflow:2.5.0-gpu-jupyter
RUN pip3 install sagemaker-training

COPY kerasclassify.py /opt/ml/code/kerasclassify.py

ENV SAGEMAKER_PROGRAM kerasclassify.py

