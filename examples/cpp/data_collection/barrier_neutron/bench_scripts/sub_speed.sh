#!/bin/bash


echoerr() { cat <<< "$@" 1>&2; }
logerr() { 
	: #cat <<< "$@" 1>&2;
}


function bceval {

scr="scratch$$"
echo "$1" | bc -l > $scr 2>&1

if grep -q error $scr
then
echoerr Error when trying to evaluate "\"$1\""
fi

cat $scr
rm -f $scr

}

function get_sum {

bceval "scale=6; $1 + $2"

}


function get_quotient {

bceval "scale=6; $1 / $2"

}

function get_product {

bceval "scale=6; $1 * $2"

}

function get_min {

bceval "scale=6; if( $1 < $2 ) $1 else $2"

}

function get_rnd {

bceval "scale=6; x=$1+0.000005; scale=5; x/1"

}

function get_pretty {

bceval "scale=1; x=$1; if(x<1) print 0; if(x==0) print \".\"; print x;"

}



argc=$#

captX=$1
scatX=$2
fissX=$3
samp=$4
sec=$5
num=$6
dev_idx=$7

size=$(( $sec + 1 ))

#max_level=134217728
max_level=33554432


hrzn_range="1 2 3 4 6 8 12 16 24 32"
#hrzn_range="1"

opt=""
suf="_s${sec}_n${num}"
tune="tune"
mark=""

if (( $argc == 8 ))
then
opt="-imp_cap -wlim $8"
mark="_imp_$8"
tune="tune_imp_$8"
elif (( $argc != 7 ))
then
echo "Bad argument count. There should be six or seven arguments."
exit
fi



superdirname="bench$suf$mark"

filename="c${captX}s${scatX}f${fissX}"




function bound_pool_size {

level=$1
check_lim=$2
approach=$3
check_command="$4"

best_time=99999999
best_size=$max_level

logerr Raw level is $level


logerr Multed level is $level

if (( $level > $max_level ))
then
logerr Maxed out
pool_size=$max_level
else
logerr Base pool size is $level
pool_size=$level
fi


canon_value=$( $check_command -pool $max_level -value )

canon_value=$( get_rnd $canon_value )

check_count=0
low_count=0
while (( 1 == 1 ))
do

logerr Trying pool size $pool_size

check_value=$( $check_command -pool $pool_size -value )
check_len=${#check_value}

if (( ( $check_len > 20 ) || ( $check_len == 0 ) ))
then
	logerr Got OOM at size $pool_size
	pool_size=$(( ( $pool_size * $approach ) / 100 ))
	break
fi

check_value=$( get_rnd $check_value )

if [[ "$check_value" != "$canon_value" ]]
then
	logerr Got bad results at size $pool_size . Check value $check_value did not equal canon value $canon_value
	pool_size=$(( ( $pool_size * $approach ) / 100 ))
	break
elif (( $pool_size <= 2048 ))
then
	pool_size=2048
	break
else
	check_count=$(( $check_count + 1 ))
	if (( $check_count >= $check_lim ))
	then
		ms=$( $check_command -pool $pool_size )
		ms=$( get_min $ms $ms )
		best_time=$( get_min $best_time $ms )

		if [[ $best_time == $ms ]]
		then
		logerr "New best time $best_time"
		best_size=$pool_size
		else
		low_count=$(( low_count + 1 ))
		if (( low_count > 5 ))
		then
			break
		fi
		logerr "Time $ms not as good as best $best_time"
		fi
		
		pool_size=$(( ( $pool_size * 100 ) / $approach ))
		check_count=0
	fi
	logerr Got okay results at size $pool_size . check count is $check_count
fi


done

check_value=$( $check_command -pool $best_size -value )
check_len=${#check_value}

if (( ( $check_len > 20 ) || ( $check_len == 0 ) ))
then
	echoerr '!!!' Got OOM at best size $best_size, which should have been okay $targ
else

check_value=$( get_rnd $check_value )

if [[ "$check_value" != "$canon_value" ]]
then
	echoerr '!!!' Got bad results $check_value vs $canon_value at best size $best_size, which should have been okay $targ
fi

fi

if (( $best_size < 2048 ))
then
	best_size=2048
fi


logerr $pool_size:$best_size

echo $pool_size:$best_size

}




function do_bench {

exe=$1
wg_count=$2
pool_mult=$3
sweep_range="$4"


dirname="$exe"

targ="$superdirname/$dirname/$filename"


rm -f "${targ}_avg"
rm -f "${targ}_avg_tune"

rm -f "${targ}_bst"
rm -f "${targ}_bst_tune"

rm -f "${targ}_mem"
rm -f "${targ}_mem_low"
rm -f "${targ}_mem_low"


minAvgTime=9999999
avgTune=9999999
minBstTime=9999999
bstTune=9999999
poolUsed=9999999


#hrzn_range=$( cat $tune/$dirname/${filename}_bst_tune )

#tune_scatX=$( get_sum $captX $scatX )
#tune_scatX=$( get_pretty $tune_scatX )
#tune_captX=0.0
#sweep_range=$( cat bench_data/heavy_data/bench${suf}/${exe}/c${tune_captX}s${tune_scatX}f${fissX}_bst_tune )







for hrzn in $sweep_range
do

if (( 1 == 1 ))
then

mem_command="${exe}_level -cx $captX -sx $scatX -fx $fissX -res 0.1 -size $size   -time $sec -hrzn $hrzn  -num $num -wg_count $wg_count -mult 2 $opt -dev_idx $dev_idx -pool $max_level"
#mem_command="neut_hrm_level -cx $captX -sx $scatX -fx $fissX -res 0.1 -size $size   -time $sec -hrzn $hrzn  -num $num -wg_count $wg_count -mult 2 $opt"
level=$( $mem_command 2>&1 )

level_len=${#level}

else

echo level= tail -n 1 bench_data/light_data/bench${suf}/${exe}/c${captX}s${scatX}f${fissX}_mem

level=$( tail -n 1 bench_data/light_data/bench${suf}/neut_hrm/c${captX}s${scatX}f${fissX}_mem )
level_len=${#level}

echo level is $level

if (( $level * 2 < $max_level ))
then
max_level=$(( $level * 2 ))
fi

fi


if (( ( $level_len > 20 ) || ( $level_len == 0 ) ))
then
echo "Could not get config for $targ to work even at max memory capacity. Got output '$level'"
exit 1
fi

check_command="$exe -cx $captX -sx $scatX -fx $fissX -res 0.1 -size $size   -time $sec -hrzn $hrzn  -num $num -wg_count $wg_count -mult 2 $opt -dev_idx $dev_idx"

if (( 1 == 1 ))
then


pool_size=$(( $level * $pool_mult ))
pool_pair=$( bound_pool_size $pool_size 2 200 "$check_command" )
echo "1) $targ -- $pool_pair -> $best_size "
best_size=${pool_pair##*:}
echo "2) $targ -- $pool_pair -> $best_size "
best_size=$(( $best_size * 2 ))
pool_size=${pool_pair%%:*}
pool_pair=$( bound_pool_size $best_size 4 150 "$check_command" )
best_size=${pool_pair##*:}
pool_size=${pool_pair%%:*}
best_size=$(( ( $best_size * 3 ) / 2 ))
pool_pair=$( bound_pool_size $best_size 6 110 "$check_command" )
best_size=${pool_pair##*:}
pool_size=${pool_pair%%:*}

else

best_size=$max_level
pool_size=$max_level

fi


echo "$hrzn:	$pool_size"  | tee -a "${targ}_mem_low"
echo "$hrzn:	$best_size"  | tee -a "${targ}_mem_bst"
pool_size=$best_size

#echo found lower pool size $pool_size







count=0
totalTime=0

bestTime=9999999

#for XX in $( seq 1 $samp )
while (( $count < $samp ))
do


perf_command="$exe -cx $captX -sx $scatX -fx $fissX -res 0.1 -size $size   -time $sec -hrzn $hrzn  -num $num -wg_count $wg_count -mult 2 $opt -pool $pool_size -dev_idx $dev_idx"
#echo "$perf_command"
result=$( $perf_command )

len=${#result}

if (( ( $len > 20 )  || ( $len <= 2 ) ))
then
echo got bad result \"$result\" when running \"$perf_command\"
pool_size=$(( ( $pool_size * 11 ) / 10 ))
continue
fi

totalTime=$( get_sum $totalTime $result )
count=$(( $count + 1 ))

bestTime=$( get_min $bestTime $result )

done

averageTime=$( get_quotient $totalTime $count )

echo "$hrzn:	$averageTime"  | tee -a "${targ}_avg"
echo "$hrzn:	$bestTime"     | tee -a "${targ}_bst"


avgTime=$( get_product $averageTime 1.02 )
minAvgTime=$( get_min $minAvgTime $avgTime )

if [[ $minAvgTime == $avgTime ]]
then
avgTune=$hrzn
minAvgTime=$averageTime
fi


bstTime=$( get_product $bestTime 1.02 )
minBstTime=$( get_min $minBstTime $bstTime )

if [[ $minBstTime == $bstTime ]]
then
bstTune=$hrzn
minBstTime=$bestTime
poolUsed=$pool_size
fi


done


echo "$minAvgTime" | tee -a "${targ}_avg"
echo "$avgTune"    | tee -a "${targ}_avg_tune"

echo "$minBstTime" | tee -a "${targ}_bst"
echo "$bstTune"    | tee -a "${targ}_bst_tune"

echo "$poolUsed"   | tee -a "${targ}_mem"


}

do_bench neut_hrm 1024 10 "1"
do_bench neut_evt 1024 10 "16" #1 2 3 4 6 8 12 16 24 32"


echo "$filename" completed

