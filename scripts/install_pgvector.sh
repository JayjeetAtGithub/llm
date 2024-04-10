#!/bin/bash
set -ex

sudo apt update
sudo apt-get install libicu-dev gcc build-essential libreadline-dev


if [ ! -d "postgres" ]; then
    git clone --branch REL_16_2 https://github.com/postgres/postgres
fi

cd postgres/

./configure --enable-debug
make -j$(nproc)

sudo make install
sudo adduser postgres
sudo mkdir -p /mnt/workspace/pgsql/data
sudo chown postgres /mnt/workspace/pgsql/data
su - postgres
/usr/local/pgsql/bin/initdb -D /mnt/workspace/pgsql/data
/usr/local/pgsql/bin/pg_ctl -D /mnt/workspace/pgsql/data -l logfile start
/usr/local/pgsql/bin/createdb test
/usr/local/pgsql/bin/psql test
