FROM nvcr.io/nvidia/cuda:12.3.1-runtime-ubuntu20.04

ADD requirements.txt /tmp
RUN apt update && apt install -y python3 python3-pip
RUN pip install -r /tmp/requirements.txt
