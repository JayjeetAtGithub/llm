#!/bin/bash
set -ex

workspace=$1

pkill -f postgres || true

sudo apt update
sudo apt-get install libicu-dev gcc build-essential libreadline-dev

if [ ! -d "postgres" ]; then
    git clone --branch REL_16_2 https://github.com/postgres/postgres
fi

if [ ! -d "pgvector" ]; then
    git clone --branch v0.7.1 https://github.com/pgvector/pgvector
fi

cd postgres/

./configure --enable-cassert --enable-debug CFLAGS="-ggdb -Og -g3 -fno-omit-frame-pointer"
sudo make -j$(nproc) install

sudo mkdir -p ${workspace}/pgsql/data
sudo chown noobjc ${workspace}/pgsql/data

cd ../pgvector/
sudo PG_CONFIG=/usr/local/pgsql/bin/pg_config make -j$(nproc) install

cd ../

# Manual steps
# /usr/local/pgsql/bin/initdb -D /mnt/workspace/pgsql/data
# /usr/local/pgsql/bin/pg_ctl -D /mnt/workspace/pgsql/data -l logfile start
# /usr/local/pgsql/bin/createdb vectordb
# /usr/local/pgsql/bin/psql vectordb
