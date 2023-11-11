
#use if on DGX or other OSU GPU systems
#CUDA_NVCC	=	/usr/local/apps/cuda/cuda-10.1/bin/nvcc

#otherwise, use this
CUDA_NVCC	=	nvcc
HIPCC	    = 	hipcc


HARMONIZE_PATH  =       ../../../harmonize.h
UTIL_PATH       =       ../../../util/util.h
UTIL_SOURCE     =       $(shell find ../../../util -type f -name '*.h' )

all: clz_hrm clz_std

clean:
	rm -f clz_hrm
	rm -f clz_std

clz_hrm: collaz_hrm.cpp $(HARMONIZE_PATH) $(UTIL_SOURCE) Makefile
	$(CUDA_NVCC) -x cu -std=c++11 --restrict -o clz_hrm collaz_hrm.cpp -include $(HARMONIZE_PATH)

clz_std: collaz_std.cpp $(UTIL_SOURCE) Makefile
	$(CUDA_NVCC) -x cu -std=c++11 -o clz_std collaz_std.cpp -include $(UTIL_PATH)


clz_evt_amd: collaz_hrm.cpp $(HARMONIZE_PATH) $(UTIL_SOURCE) Makefile
	hipify-clang collaz_hrm.cpp     --default-preprocessor -- -include $(HARMONIZE_PATH) --std=c++17 -DEVENT_MODE
	$(HIPCC) -x hip -std=c++11 -o clz_evt_amd collaz_hrm.cpp.hip -include $(HARMONIZE_PATH) --std=c++17 -DEVENT_MODE

