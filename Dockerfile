FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu20.04

ADD requirements.txt /app

WORKDIR /app

RUN apt update && apt install -y python3 python3-pip
RUN pip install -r requirements.txt
