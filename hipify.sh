#!/bin/bash -e

hipify-clang harmonize.cpp --default-preprocessor --cuda-path=/usr/lib/cuda/ -- --std=c++17 -DHIPIFY=1
hipify-clang util/*.h --default-preprocessor --cuda-path=/usr/lib/cuda/ -- --std=c++17 -DHIPIFY=1

