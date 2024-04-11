#!/bin/bash
set -ex

if [ ! -d "faiss" ]; then
    git clone https://github.com/facebookresearch/faiss
fi

cd faiss/
git pull

sudo apt-get update
sudo apt-get install -y libopenblas-dev

rm -rf build/
mkdir -p build/
cd build/

cmake -DFAISS_ENABLE_GPU=OFF \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DFAISS_OPT_LEVEL=avx2 \
      -DBUILD_TESTING=OFF \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-ggdb -Og -g3 -fno-omit-frame-pointer" \
      -DCMAKE_C_FLAGS="-ggdb -Og -g3 -fno-omit-frame-pointer"
     ..

sudo make -j$(nproc)
sudo make install
