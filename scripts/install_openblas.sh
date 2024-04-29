#!/bin/bash
set -ex

workspace=$1

if [ ! -d "${workspace}/OpenBLAS" ]; then
    git clone https://github.com/OpenMathLib/OpenBLAS ${workspace}/OpenBLAS
fi

cd ${workspace}/OpenBLAS
git pull

cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-ggdb -Og -g3 -fno-omit-frame-pointer" \
      -DCMAKE_C_FLAGS="-ggdb -Og -g3 -fno-omit-frame-pointer" \
      .

sudo make -j$(nproc) install PREFIX=/usr/local

