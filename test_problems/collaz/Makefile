CUDA_PATH	=	/usr/local/apps/cuda/cuda-10.1
CUDA_BIN_PATH	=	$(CUDA_PATH)/bin
CUDA_NVCC	=	$(CUDA_BIN_PATH)/nvcc

all: clz_hrm clz_std

hard: do_touch all

do_touch:
	touch collaz.cu
	touch vanilla_collaz.cu

clz_hrm: collaz_hrm.cu ../../harmonize.cu ../../util.cu
	$(CUDA_NVCC) -std=c++11 --restrict -o clz_hrm collaz_hrm.cu

clz_std: collaz_std.cu ../../util.cu
	$(CUDA_NVCC) -std=c++11 -o clz_std collaz_std.cu
