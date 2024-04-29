#!/bin/bash
set -ex

workspace=$1

cd ${workspace}
rm -rf cmake-3.29.2*
wget https://github.com/Kitware/CMake/releases/download/v3.29.2/cmake-3.29.2.tar.gz
tar -xvzf cmake-3.29.2.tar.gz

cd cmake-3.29.2
./bootstrap
sudo make -j$(nproc) install
