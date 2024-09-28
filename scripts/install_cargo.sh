#!/usr/bin/env sh
set -e

pip install boto3

apt update
apt-get install libssl-dev -y

# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

. "$HOME/.cargo/env"

rustup install nightly
rustup update nightly
rustup override set nightly

cargo update

export RUSTFLAGS="-C target-cpu=native -C linker=gcc"

cargo install cargo-simd-detect --force
cargo simd-detect

