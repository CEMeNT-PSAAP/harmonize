#!/bin/bash
#SBATCH -J harmony_bench
#SBATCH -A eecs
#SBATCH -p dgx2
#SBATCH --gres=gpu:1
#SBATCH -o harmony_debug.out
#SBATCH -e harmony_debug.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cuneob@oregonstate.edu


#make hard SM_COUNT=2 SPAN=1024 QUEUE_WIDTH=64 ARENA_SIZE=0xFFFFF
#make hard SM_COUNT=16 SPAN=65536 QUEUE_WIDTH=64 ARENA_SIZE=0xFFFFF
#make hard SM_COUNT=16 SPAN=131072 QUEUE_WIDTH=64 ARENA_SIZE=0xFFFFF
#make hard SM_COUNT=16 SPAN=262144 QUEUE_WIDTH=64 ARENA_SIZE=0xFFFFF
#make hard SM_COUNT=16 SPAN=524288 QUEUE_WIDTH=64 ARENA_SIZE=0xFFFFF
#make hard SM_COUNT=16 SPAN=1048576 QUEUE_WIDTH=64 ARENA_SIZE=0xFFFFF

#make hard SM_COUNT=256 SPAN=1048576 QUEUE_WIDTH=64 ARENA_SIZE=0xFFFFF
#make hard SM_COUNT=256 SPAN=16777216 QUEUE_WIDTH=64 ARENA_SIZE=0xFFFFF
make hard SM_COUNT=256 SPAN=33554432 QUEUE_WIDTH=64 ARENA_SIZE=0xFFFFF
#make hard SM_COUNT=256 SPAN=67108864 QUEUE_WIDTH=64 ARENA_SIZE=0xFFFFF

mkdir -p ./dbg_out

#collaz_hrm_split > ./split_out

for i in $(seq 0 4)
do
	collaz_hrm_unified #>  ./dbg_out/unif_dbg_$i	
	echo               #>> ./dbg_out/unif_dbg_$i

done

#for i in $(seq 0 4)
#do
#	collaz_hrm_split   #>  ./dbg_out/splt_dbg_$i	
#	echo               #>> ./dbg_out/splt_dbg_$i
#
#done


for i in $(seq 0 4)
do
	collaz_van
	echo

done

#collaz_van > ./vanilla_out






