#!/bin/bash
set -ex

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
./bin/profile_faiss flat gist index 10 debug
./bin/profile_faiss hnsw gist index 10 debug
./bin/profile_hnswlib flat gist index 10 debug
./bin/profile_hnswlib hnsw gist index 10 debug
