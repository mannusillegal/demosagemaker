FROM ubuntu
FROM  continuumio/miniconda3:latest
RUN apt-get update && apt-get install -
RUN pip install tensorflow==2.5.1

copy . /ml/
WORKDIR /ml
CMD python kerasclassify.py