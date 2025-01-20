#!/bin/bash

# Ensure required tools are installed
sudo apt-get update
sudo apt-get install -y build-essential libreadline-dev

# Download and build SQLite 3.45.3 (or another version)
SQLITE_VERSION=3.45.3
wget https://www.sqlite.org/2024/sqlite-autoconf-${SQLITE_VERSION//./}.tar.gz
tar -xzf sqlite-autoconf-${SQLITE_VERSION//./}.tar.gz
cd sqlite-autoconf-${SQLITE_VERSION//./}
./configure --prefix=$HOME/.local
make -j$(nproc)
make install

# Update PATH and library paths
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

# Confirm installation
sqlite3 --version
