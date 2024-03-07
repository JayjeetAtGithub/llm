#!/bin/bash
set -ex

# Remove any previous installation
rm -rf milvus_2.3.10-1_amd64.deb
sudo dpkg -P milvus

# Download and install Milvus
wget https://github.com/milvus-io/milvus/releases/download/v2.3.10/milvus_2.3.10-1_amd64.deb
sudo apt-get update
sudo dpkg -i milvus_2.3.10-1_amd64.deb
sudo apt-get -f install

# Start Milvus
sudo systemctl restart milvus
sudo systemctl status milvus
