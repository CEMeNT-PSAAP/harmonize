CUDA_PATH	=	/usr/local/apps/cuda/cuda-10.1
CUDA_BIN_PATH	=	$(CUDA_PATH)/bin
CUDA_NVCC	=	$(CUDA_BIN_PATH)/nvcc

HARMONIZE_PATH  =       ../../harmonize.cu
UTIL_PATH       =       ../../util/util.cu
UTIL_SOURCE     =       $(shell find ../../util -type f -name '*.cu' )

all: clz_hrm clz_std

hard: do_touch all

do_touch:
	touch collaz.cu
	touch vanilla_collaz.cu

clz_hrm: collaz_hrm.cu $(HARMONIZE_PATH) $(UTIL_SOURCE)
	$(CUDA_NVCC) -std=c++11 --restrict -o clz_hrm collaz_hrm.cu -include $(HARMONIZE_PATH)

clz_std: collaz_std.cu $(UTIL_SOURCE)
	$(CUDA_NVCC) -std=c++11 -o clz_std collaz_std.cu -include $(UTIL_PATH)
