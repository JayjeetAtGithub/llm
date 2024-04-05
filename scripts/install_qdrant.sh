#!/bin/bash
set -ex

# setup dependencies: clang, protobuf
sudo apt-get update
sudo apt-get install -y libclang-dev

wget https://github.com/protocolbuffers/protobuf/releases/download/v26.1/protoc-26.1-linux-x86_64.zip
unzip protoc-26.1-linux-x86_64.zip
cp -r bin/* /usr/local/bin/
cp -r include/* /usr/local/include/

# download qdrant
if [ ! -d "qdrant" ]; then
    git clone https://github.com/qdrant/qdrant
fi

cd qdrant/
git pull

cargo build --release
