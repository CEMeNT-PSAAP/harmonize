#!/bin/bash



for i in $(seq 0 $1)
do
	grep "($i)" test_out | rev | cut -d' ' -f 1 | rev | paste -sd+ | bc > delta_left
	grep "SM $i " test_out | rev | cut -d' ' -f 1 | rev > delta_right
	left=$(cat delta_left)
	right=$(cat delta_right)
	if [ "$left" != "$right" ]
	then
		echo -e "Mismatch for SM $i:\tper-delta count: $left\treported total: $right"
		grep "($i)" test_out
	fi
done

rm -f delta_left
rm -f delta_right


