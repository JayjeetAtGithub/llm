#!/bin/bash
set -ex

sudo apt update
sudo apt-get install libicu-dev gcc build-essential libreadline-dev


if [ ! -d "postgres" ]; then
    git clone --branch REL_16_2 https://github.com/postgres/postgres
fi

if [ ! -d "pgvector" ]; then
    git clone --branch v0.6.2 https://github.com/pgvector/pgvector
fi

cd postgres/

./configure --enable-debug
sudo make -j$(nproc) install

sudo adduser postgres
sudo mkdir -p /mnt/workspace/pgsql/data
sudo chown postgres /mnt/workspace/pgsql/data

cd ../pgvector/
sudo PG_CONFIG=/usr/local/pgsql/bin/pg_config make -j$(nproc) install

cd ../

# manual steps
# su - postgres
# /usr/local/pgsql/bin/initdb -D /mnt/workspace/pgsql/data
# /usr/local/pgsql/bin/pg_ctl -D /mnt/workspace/pgsql/data -l logfile start
# /usr/local/pgsql/bin/createdb vectordb
# /usr/local/pgsql/bin/psql vectordb
