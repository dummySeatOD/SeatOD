#!/usr/bin/env bash

if ! [ -d "models_stereo/result/data" ]; then
    echo "Results not found in models_stereo/result/data"
    exit
else
    rm -rf models_stereo/result/plot
    rm -f models_stereo/result/stats*.txt
    rm -f models_stereo/result/AP_result.txt
    mkdir models_stereo/result/plot
fi
./eval_tool/build/evaluate_object