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
    git clone --branch v1.8.4 https://github.com/qdrant/qdrant
fi

cd qdrant/
git pull

# for some reason, building qdrant in debug mode does not give any profile information, weird !!!
cargo build --release
