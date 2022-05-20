
#use if on DGX or other OSU GPU systems
#CUDA_NVCC	=	/usr/local/apps/cuda/cuda-10.1/bin/nvcc

#otherwise, use this
CUDA_NVCC	=	nvcc

FLAGS		=	-std=c++11 --compiler-options -fdiagnostics-color=always -g

HARMONIZE_PATH  =       ../../harmonize.cpp
UTIL_PATH       =       ../../util/util.cpp
UTIL_SOURCE     =       $(shell find ../../util -type f -name '*.cpp' )


ifndef ARCH
ARCH_DEF=
else
ARCH_DEF=-arch=$(ARCH)
endif


all: demo


clean:
	rm -f demo

demo:      Makefile $(HARMONIZE_PATH) demo.cpp $(UTIL_SOURCE)
	$(CUDA_NVCC) -x cu $(ARCH_DEF) $(FLAGS) -o demo demo.cpp --restrict -include $(HARMONIZE_PATH)

