#!/bin/bash
set -ex

workspace=$1

if [ ! -d "${workspace}/faiss" ]; then
    git clone https://github.com/facebookresearch/faiss ${workspace}/faiss
fi

cd ${workspace}/faiss/
git pull

rm -rf build/
mkdir -p build/
cd build/

cmake -DFAISS_ENABLE_GPU=OFF \
      -DFAISS_ENABLE_PYTHON=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DFAISS_OPT_LEVEL=avx2 \
      -DBUILD_TESTING=OFF \
      -DCMAKE_BUILD_TYPE=Release \
     ..

sudo make -j$(nproc)
sudo make install
