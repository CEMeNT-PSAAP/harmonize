#!/bin/bash

rm -f data.out

samp_count=32

for (( inp = 1; inp <= 1048576; inp *= 4 ))
do

for (( lim=1; lim<=256; lim*=2 ))
do

avg=0
sum="scale=6; ( 0"

for (( samp=0; samp < samp_count; samp++ ))
do

val=$( ./demo -wg_count 1024 -i_size $inp -o_size 65536 -limit $lim )
sum="$sum + $val"

done

#echo "$sum ) / $samp_count "

echo "$sum ) / $samp_count " | bc > scratch$$

avg=$( cat scratch$$ )

echo -n $avg | tee -a data.out
echo -n -e "\t" | tee -a data.out

done

echo | tee -a data.out

done


