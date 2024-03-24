#!/bin/bash
set -ex

git clone https://github.com/facebookresearch/faiss
cd faiss/

sudo apt-get install -y libopenblas-dev

rm -rf build/
mkdir -p build/
cd build/

cmake -DFAISS_ENABLE_GPU=OFF \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DFAISS_OPT_LEVEL=avx2 \
     ..

sudo make -j$(nproc)
