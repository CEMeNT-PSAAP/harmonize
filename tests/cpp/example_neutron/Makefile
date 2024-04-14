
#use if on DGX or other OSU GPU systems
#CUDA_NVCC	=	/usr/local/apps/cuda/cuda-10.1/bin/nvcc

#otherwise, use this
CUDA_NVCC	=	nvcc

FLAGS		=	-std=c++11 --compiler-options -fdiagnostics-color=always -g

HARMONIZE_PATH  =       ../../../include/harmonize.h


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

