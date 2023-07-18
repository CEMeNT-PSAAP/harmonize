
#use if on DGX or other OSU GPU systems
#CUDA_NVCC	=	/usr/local/apps/cuda/cuda-10.1/bin/nvcc

#otherwise, use this
CUDA_NVCC	=	nvcc


HARMONIZE_PATH  =       ../../../harmonize.cpp
UTIL_PATH       =       ../../../util/util.cpp
UTIL_SOURCE     =       $(shell find ../../../util -type f -name '*.cpp' )

all: clz_hrm clz_std

clean:
	rm -f clz_hrm
	rm -f clz_std

clz_hrm: collaz_hrm.cpp $(HARMONIZE_PATH) $(UTIL_SOURCE)
	$(CUDA_NVCC) -x cu -std=c++11 --restrict -o clz_hrm collaz_hrm.cpp -include $(HARMONIZE_PATH)

clz_std: collaz_std.cpp $(UTIL_SOURCE)
	$(CUDA_NVCC) -x cu -std=c++11 -o clz_std collaz_std.cpp -include $(UTIL_PATH)
