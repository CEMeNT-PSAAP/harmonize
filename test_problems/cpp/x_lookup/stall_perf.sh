#!/bin/bash


sum_like () {
    last_name=""
    total=0
    while read name unit value
    do
        if [ "$name" != "$last_name" ] && [ "$last_name" != "" ]
        then
            echo -e "$last_name\t$total"
            total=0
        fi
        value=$( echo $value | tr -d ',' )
        total=$(( total + value ))
        last_name=$name
    done
    echo $last_name $total
}


echo "Async Stall Metrics"
# 240 WG is typically optimal for async, based on testing
ncu  --metrics 'regex:stalled' ./neut.exe -wg_count 240 -num 10000000 -pool 10000000 -cross_count 100000000 -time 20 -size 20 -res 0.01 -show -fx_max 0.4 -sx_max 0.2 -cx_max 0.4 -async |
    grep '\.sum ' | sort | sum_like


echo "Event Stall Metrics"
# >1024 WG is typically optimal for sync, based on testing
ncu  --metrics 'regex:stalled' ./neut.exe -wg_count 1024 -num 10000000 -pool 10000000 -cross_count 100000000 -time 20 -size 20 -res 0.01 -show -fx_max 0.4 -sx_max 0.2 -cx_max 0.4 |
    grep '\.sum ' | sort | sum_like


