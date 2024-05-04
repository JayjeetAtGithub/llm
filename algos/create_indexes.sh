#!/bin/bash
set -ex

./bin/profile_faiss flat gist index 10 debug
./bin/profile_faiss hnsw gist index 10 debug
./bin/profile_hnswlib flat gist index 10 debug
./bin/profile_hnswlib hnsw gist index 10 debug
