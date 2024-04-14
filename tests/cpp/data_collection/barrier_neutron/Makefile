
#use if on DGX or other OSU GPU systems
#CUDA_NVCC	=	/usr/local/apps/cuda/cuda-10.1/bin/nvcc

#otherwise, use this
CUDA_NVCC	=	nvcc

FLAGS		=	-std=c++11 --compiler-options -fdiagnostics-color=always -g

HARMONIZE_PATH  =       ../../../../include/harmonize.h


ifndef ARCH
ARCH_DEF=
else
ARCH_DEF=-arch=$(ARCH)
endif


neut:      Makefile $(HARMONIZE_PATH) neut_harmonized.cpp neut_common.cpp
	$(CUDA_NVCC) -x cu -DCACHE $(ARCH_DEF)  $(FLAGS) neut_harmonized.cpp -o neut --restrict -include $(HARMONIZE_PATH)  --expt-relaxed-constexpr

clean:
	rm -f neut


