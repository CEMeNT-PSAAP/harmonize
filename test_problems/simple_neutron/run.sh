#!/bin/bash

srun -J hrm_job -A eecs -p dgx2 --gres=gpu:1 --mail-type=NONE $@


