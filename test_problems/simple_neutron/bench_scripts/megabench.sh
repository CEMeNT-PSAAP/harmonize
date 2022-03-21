#!/bin/bash


for sec in  2 4 6 #8
do

for num in  1000 10000 100000 #1000000
do

quickbench.sh $sec $num

done

done

