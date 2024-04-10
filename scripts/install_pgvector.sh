#!/bin/bash
set -ex

if [ ! -d "postgres" ]; then
    git clone --branch REL_16_2 https://github.com/postgres/postgres
fi

cd postgres/

./configure --enable-debug
make

sudo make install
sudo adduser postgres
sudo mkdir -p /usr/local/pgsql/data
sudo chown postgres /usr/local/pgsql/data
su - postgres
/usr/local/pgsql/bin/initdb -D /mnt/workspace/pgsql/data
/usr/local/pgsql/bin/pg_ctl -D /mnt/workspace/pgsql/data -l logfile start
/usr/local/pgsql/bin/createdb test
/usr/local/pgsql/bin/psql test
