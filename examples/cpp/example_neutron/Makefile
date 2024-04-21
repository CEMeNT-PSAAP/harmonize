# Use this on lassen
# CUDA_NVCC     = /usr/tce/packages/cuda/cuda-11.5.0/bin/nvcc

CUDA_NVCC	=	nvcc
FLAGS		=	-std=c++11 --compiler-options -fdiagnostics-color=always -g
HARMONIZE_PATH  = ../../../harmonize/cpp/harmonize.h


ifndef ARCH
ARCH_DEF=
else
ARCH_DEF=-arch=$(ARCH)
endif


all: neut.exe


clean:
	rm -f neut.exe

neut.exe:      Makefile $(HARMONIZE_PATH) program.cpp logic.cpp
	$(CUDA_NVCC) -x cu $(ARCH_DEF) $(FLAGS) -o neut.exe program.cpp --restrict -include $(HARMONIZE_PATH)

