#!/bin/bash
set -ex

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
./bin/profile_faiss flat gist query 10 debug
./bin/profile_faiss hnsw gist query 10 debug
./bin/profile_hnswlib flat gist query 10 debug
./bin/profile_hnswlib hnsw gist query 10 debug

OMP_NUM_THREADS=1 ./bin/profile_faiss flat gist query 10 debug
OMP_NUM_THREADS=1 ./bin/profile_faiss hnsw gist query 10 debug
OMP_NUM_THREADS=1 ./bin/profile_hnswlib flat gist query 10 debug
OMP_NUM_THREADS=1 ./bin/profile_hnswlib hnsw gist query 10 debug