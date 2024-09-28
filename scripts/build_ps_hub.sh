#!/usr/bin/env sh
set -e

. "$HOME/.cargo/env"

cargo build --release
