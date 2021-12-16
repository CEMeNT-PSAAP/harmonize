CUDA_PATH	=	/usr/local/apps/cuda/cuda-10.1
CUDA_BIN_PATH	=	$(CUDA_PATH)/bin
CUDA_NVCC	=	$(CUDA_BIN_PATH)/nvcc
STD		=	-std=c++11

HARMONIZE_PATH  =       ../../harmonize.cpp
UTIL_PATH       =       ../../util/util.cpp
UTIL_SOURCE     =       $(shell find ../../util -type f -name '*.cpp' )


ARCH=sm_50

all: neut_hrm neut_std neut_evt


clean:
	rm -f neut_evt
	rm -f neut_hrm
	rm -f neut_std
	rm -f neuttest.out

neut_hrm:      Makefile $(HARMONIZE_PATH) neut_harmonized.cpp neut_common.cpp $(UTIL_SOURCE)
	$(CUDA_NVCC) -x cu -arch=$(ARCH) -DCACHE $(STD) -o neut_hrm neut_harmonized.cpp --restrict -include $(HARMONIZE_PATH)


neut_std:      Makefile neut_standard.cpp neut_common.cpp $(UTIL_SOURCE)
	$(CUDA_NVCC) -x cu -arch=$(ARCH) $(STD) -o neut_std neut_standard.cpp --restrict -include $(UTIL_PATH)


neut_evt:      Makefile $(HARMONIZE_PATH) neut_harmonized.cpp neut_common.cpp $(UTIL_SOURCE)
	$(CUDA_NVCC) -x cu -arch=$(ARCH) $(STD) -o neut_evt neut_harmonized.cpp -DEVENT --restrict -include $(HARMONIZE_PATH)

