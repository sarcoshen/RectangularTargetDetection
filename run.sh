#!/bin/bash

rm -rf main
rm -rf CMakeCache.txt cmake_install.cmake CMakeFiles Makefile
cmake .
make
./main
