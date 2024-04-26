#!/bin/bash
set -ex

rm -rf /mnt/workspace/cmake-3.29.2*
wget https://github.com/Kitware/CMake/releases/download/v3.29.2/cmake-3.29.2.tar.gz -O /mnt/workspace/cmake-3.29.2.tar.gz
tar -xvzf /mnt/workspace/cmake-3.29.2.tar.gz

cd /mnt/workspace/cmake-3.29.2
./bootstrap
sudo make -j$(nproc) install
