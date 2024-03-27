#!/bin/sh
# compile script

cd example || exit

if [ ! -d "build" ]; then
  mkdir build
fi

cd build || exit
cmake .. && make