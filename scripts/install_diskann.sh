#!/bin/bash
set -ex

# install dependencies
sudo apt-get update
sudo apt-get -y install make \
                        cmake \
                        g++ \
                        libaio-dev \
                        libgoogle-perftools-dev \
                        clang-format \
                        libboost-all-dev \
                        libomp-dev

# install intel mkl


# clone diskann code
if [ ! -d "/mnt/workspace/diskann" ]; then
    git clone https://github.com/microsoft/diskann /mnt/workspace/diskann
fi

cd /mnt/workspace/diskann
git pull

# build diskann
mkdir build/
cd build/
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

