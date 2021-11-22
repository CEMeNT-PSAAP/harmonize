#!/bin/bash

#SBATCH  -J  fisstest
#SBATCH  -A  eecs
#SBATCH  -p  dgx2
#SBATCH  --gres=gpu:1
#SBATCH  -o  fisstest.out
#SBATCH  -e  fisstest.err
#SBATCH  --mail-type=NONE
#SBATCH  --mail-user=cuneob@oregonstate.edu

$@


