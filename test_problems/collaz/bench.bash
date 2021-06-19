#!/bin/bash
#SBATCH -J harmony_bench
#SBATCH -A cs475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o harmony_bench.out
#SBATCH -e harmony_bench.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cuneob@oregonstate.edu

rm -f ./split_out
rm -f ./unified_out
rm -f ./vanilla_out

touch ./split_out
touch ./unified_out
touch ./vanilla_out


sample_count=16

#for span in 65536 100000
for span in 1024 2048 4096 8192 16384 32768 65536 100000
do

#for sm_count in $(seq 1 3)
for sm_count in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
do

make hard SM_COUNT=$sm_count SPAN=$span

split_agg=0
unified_agg=0
vanilla_agg=0

for sample in $(seq 1 $sample_count)
do

split_val=$(collaz_hrm_split)
unified_val=$(collaz_hrm_unified)
vanilla_val=$(collaz_van)

echo "split_val: $split_val"
echo "unified_val: $unified_val"
echo "vanilla_val: $vanilla_val"

echo "scale=6; $split_val + $split_agg" | bc > ./bench_temp
split_agg=$(cat ./bench_temp)

echo "scale=6; $unified_val + $unified_agg" | bc > ./bench_temp
unified_agg=$(cat ./bench_temp)

echo "scale=6; $vanilla_val + $vanilla_agg" | bc > ./bench_temp
vanilla_agg=$(cat ./bench_temp)

done

echo "scale=6; $split_agg / $sample_count.0000"   | bc | tr '\n' '\t' >> ./split_out
echo "scale=6; $unified_agg / $sample_count.0000" | bc | tr '\n' '\t' >> ./unified_out
echo "scale=6; $vanilla_agg / $sample_count.0000" | bc | tr '\n' '\t' >> ./vanilla_out

echo "Split:"
cat ./split_out
echo

echo "Unified:"
cat ./unified_out
echo

echo "Vanilla:"
cat ./vanilla_out
echo

done

echo >> ./split_out
echo >> ./unified_out
echo >> ./vanilla_out

done





