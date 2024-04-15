#!/bin/bash
set -ex

# install dependencies
sudo apt-get update
sudo apt-get -y install make \
                        cmake \
                        g++ \
                        gpg-agent \
                        wget \
                        libaio-dev \
                        libgoogle-perftools-dev \
                        clang-format \
                        libboost-all-dev \
                        libomp-dev

# install intel mkl
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update
sudo apt install intel-oneapi-mkl-devel

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
