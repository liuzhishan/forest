#!/bin/bash

source ./util.sh
root=$(dirname `pwd`)

if [ ! -f $root/target/release/hub-server ]; then
    echo "Error: hub-server not found. Please execute ./scripts/build_ps_hub.sh under sniper root path first."
    exit 1
fi

if [ ! -f ./hub_server ]; then
    ln -s $root/target/release/hub-server ./hub_server
fi

mkdir -p /data/ad/log
./hub_server > /data/ad/log/hub.log 2>&1