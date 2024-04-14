#!/bin/bash

argc=$#

if   (( argc == 3 ))
then
	secs=$1
	num=$2
	wlim=$3
	dirname="bench_s${secs}_n${num}_imp_${wlim}"
elif (( argc == 2 ))
then
	secs=$1
	num=$2
	dirname="bench_s${secs}_n${num}"
else
	echo "Wrong number of arguments. Should be three or four (<time horizon> <source particle count> <offset> [wlim])"
	exit 1
fi


exelist="neut_hrm neut_evt"

mkdir -p "$dirname"

for exe in $exelist
do

mkdir -p "$dirname/$exe"

done



super_limit=9
limit=4
#super_limit=66
#limit=1

super_count=0
count=0
cmd=""


function gen_name {

captX=$1
scatX=$2
fissX=$3

echo "c${captX}s${scatX}f${fissX}"

}



function bceval {

scr="scratch$$"
echo "$1" | bc -l > $scr
cat $scr
rm -f $scr

}

function norm {

bceval "scale=1; x = $1 / 10; if(x<1) print 0; if(x==0) print \".\"; print x;"

}





#for fission in $( seq 0 10 )
#do
#scatterLim=$(( 10 - fission ))
#for scatter in $( seq 0 $scatterLim )
#do
#capture=$(( 10 - fission - scatter ))


for capture in $( seq 0 10 )
do

fissionLim=$(( 10 - capture ))

for fission in $( seq 0 $fissionLim )
do

scatter=$(( 10 - fission - capture ))

#if (( $capture < fission ))
#then
#break
#fi

fissX=$( norm $fission )
scatX=$( norm $scatter )
captX=$( norm $capture )

#samp=32
samp=128


name=$(gen_name $captX $scatX $fissX)

#if [[ ! -f "${dirname}/neut_evt/${name}_avg" || ! -f "${dirname}/neut_hrm/${name}_avg" ]]
if [[ ! -f "${dirname}/neut_hrm/${name}_avg" ]]
then
#srun -N 1 -n 1 sub_speed.sh $captX $scatX $fissX $samp $secs $num $count $wlim &
#jsrun -p 1 -c 1 -g 1 sub_speed.sh $captX $scatX $fissX $samp $secs $num 0 $wlim &
cmd="$cmd sub_speed.py $captX $scatX $fissX $samp $secs $num $count $wlim &"
count=$(( $count + 1 ))
fi

#               sub_speed.sh $captX $scatX $fissX $samp $secs $num $wlim



if (( $count >= $limit ))
then
srun -N 1 -n 1 go_run.sh "$cmd" &
#srun -A eecs -p dgx2 --gres=gpu:$count go_run.sh "$cmd" &
count=0
cmd=""
super_count=$(( $super_count + 1 ))
fi

if (( $super_count >= $super_limit ))
then
wait
super_count=0
fi

done

done

if (( $count > 0 ))
then
srun -N 1 -n 1 go_run.sh "$cmd" &
#srun -A eecs -p dgx2 --gres=gpu:$count go_run.sh "$cmd" &
cmd=""
fi

wait




for exe in $exelist
do


rm "$dirname/$exe/avg.log"
rm "$dirname/$exe/avg_tune.log"

rm "$dirname/$exe/bst.log"
rm "$dirname/$exe/bst_tune.log"

rm "$dirname/$exe/mem.log"

for capture in $( seq 10 -1 0 )
do

fissionLim=$(( 10 - capture ))

for fission in $( seq 0 $fissionLim )
do

scatter=$(( 10 - fission - capture ))

fissX=$( norm $fission )
scatX=$( norm $scatter )
captX=$( norm $capture )


name=$(gen_name $captX $scatX $fissX)
targ="$dirname/$exe/$name"


cat "${targ}_avg"      | tail -n 1 | tr '\n' '\t' | tee -a "$dirname/$exe/avg.log"
cat "${targ}_avg_tune" | tail -n 1 | tr '\n' '\t' | tee -a "$dirname/$exe/avg_tune.log"

cat "${targ}_bst"      | tail -n 1 | tr '\n' '\t' | tee -a "$dirname/$exe/bst.log"
cat "${targ}_bst_tune" | tail -n 1 | tr '\n' '\t' | tee -a "$dirname/$exe/bst_tune.log"

cat "${targ}_mem"      | tail -n 1 | tr '\n' '\t' | tee -a "$dirname/$exe/mem.log"

done

echo | tee -a "$dirname/$exe/avg.log"
echo | tee -a "$dirname/$exe/avg_tune.log"

echo | tee -a "$dirname/$exe/bst.log"
echo | tee -a "$dirname/$exe/bst_tune.log"

echo | tee -a "$dirname/$exe/mem.log"

done

done



