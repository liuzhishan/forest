#!/usr/bin/env sh
set -e

g++ test.cc -mavx -mavx512f -lglog -o test
./test
