#!/bin/bash
set -ex

DATASET=$1

wget ftp://ftp.irisa.fr/local/texmex/corpus/${DATASET}.tar.gz
tar -xvzf ${DATASET}.tar.gz
