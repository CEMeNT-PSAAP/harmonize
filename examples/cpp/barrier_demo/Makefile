CUDA_NVCC	=	nvcc
FLAGS		=	-std=c++11 --compiler-options -fdiagnostics-color=always -g
HARMONIZE_PATH  = ../../../include/harmonize.h


ifndef ARCH
ARCH_DEF=
else
ARCH_DEF=-arch=$(ARCH)
endif


all: demo.exe


clean:
	rm -f demo.exe

demo.exe:      Makefile $(HARMONIZE_PATH) demo.cpp $(UTIL_SOURCE)
	$(CUDA_NVCC) -x cu $(ARCH_DEF) $(FLAGS) -o demo.exe demo.cpp --restrict -include $(HARMONIZE_PATH)

