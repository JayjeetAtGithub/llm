#!/bin/bash
set -ex

pwd=$PWD

sudo apt update
sudo apt install -y python3 \
               python3-pip \
               python3-venv \
               python3-numpy \
               cmake \
               libboost-all-dev \
               libssl-dev \
               llvm

if [ ! -d "/mnt/workspace/arrow" ]; then
    git clone https://github.com/apache/arrow /mnt/workspace/arrow
fi

cd /mnt/workspace/arrow
git checkout apache-arrow-15.0.2
git submodule update --init --recursive
git pull origin apache-arrow-15.0.2

mkdir -p cpp/debug
cd cpp/debug

cmake -DARROW_PARQUET=ON \
  -DARROW_WITH_SNAPPY=ON \
  -DARROW_CSV=ON \
  -DARROW_DATASET=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-ggdb -Og -g3 -fno-omit-frame-pointer" \
  -DCMAKE_C_FLAGS="-ggdb -Og -g3 -fno-omit-frame-pointer" \
  ..

sudo make -j$(nproc) install

sudo cp -r /usr/local/lib/* /usr/lib/
cd $pwd
