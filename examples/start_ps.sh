#!/bin/bash

source ./util.sh
root=$(dirname `pwd`)

if [ ! -f $root/target/release/ps-server ]; then
    echo "Error: ps-server not found. Please execute ./scripts/build_ps_hub.sh under sniper root path first."
    exit 1
fi

if [ ! -f ./ps_server ]; then
    ln -s $root/target/release/ps-server ./ps_server
fi

mkdir -p /data/ad/log
./ps_server > /data/ad/log/ps.log 2>&1