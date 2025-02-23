#!/bin/bash

M=1
STEP=64
for ((n=$STEP; n<=16384; n+=$STEP))
do
    echo "Running with M=$M and n=$n"
    ./build/cumsum $M $n
done