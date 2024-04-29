#!/bin/bash
set -ex

workspace=$1

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

# clone DiskANN code
if [ ! -d "${workspace}/DiskANN" ]; then
    git clone https://github.com/microsoft/DiskANN ${workspace}/DiskANN
fi

cd ${workspace}/DiskANN
git pull

# build DiskANN
mkdir -p build/
cd build/

cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-ggdb -Og -g3 -fno-omit-frame-pointer" \
      -DCMAKE_C_FLAGS="-ggdb -Og -g3 -fno-omit-frame-pointer" \
      ..

make -j$(nproc)
