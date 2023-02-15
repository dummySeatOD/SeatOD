#!/usr/bin/env bash

if [ $# -eq 1 ]; then
    if [ $1 == "clean" ]; then
        echo "Clean build..."
        rm -rf lib/build
        rm -rf lib/faster_rcnn.egg-info
        rm -rf eval_tool/build
    fi
fi

cd lib
echo "Building Lib..."
python setup.py build develop
echo "###########################################"
cd ../eval_tool
echo "Building Evaluation Tool..."
if ! [ -d "./build" ]; then
    mkdir build
fi
cd build
cmake ..
make
cp evaluate_object ..
cd ../..
