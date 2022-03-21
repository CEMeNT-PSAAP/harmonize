
#use if on DGX or other OSU GPU systems
CUDA_NVCC	=	/usr/local/apps/cuda/cuda-10.1/bin/nvcc

#otherwise, use this
#CUDA_NVCC	=	nvcc

FLAGS		=	-std=c++11 --compiler-options -fdiagnostics-color=always -g

HARMONIZE_PATH  =       ../../harmonize.cpp
UTIL_PATH       =       ../../util/util.cpp
UTIL_SOURCE     =       $(shell find ../../util -type f -name '*.cpp' )


ifndef ARCH
ARCH_DEF=
else
ARCH_DEF=-arch=$(ARCH)
endif


all: neut_hrm neut_evt neut_hrm_level neut_evt_level


clean:
	rm -f neut_evt
	rm -f neut_evt_level
	rm -f neut_hrm
	rm -f neut_hrm_level
	rm -f neuttest.out

neut_hrm:      Makefile $(HARMONIZE_PATH) neut_harmonized.cpp neut_common.cpp $(UTIL_SOURCE)
	$(CUDA_NVCC) -x cu -DCACHE $(ARCH_DEF)  $(FLAGS) -o neut_hrm neut_harmonized.cpp --restrict -include $(HARMONIZE_PATH)

neut_hrm_level:      Makefile $(HARMONIZE_PATH) neut_harmonized.cpp neut_common.cpp $(UTIL_SOURCE)
	$(CUDA_NVCC) -x cu -DCACHE -DLEVEL_CHECK $(ARCH_DEF)  $(FLAGS) -o neut_hrm_level neut_harmonized.cpp --restrict -include $(HARMONIZE_PATH)



neut_std:      Makefile neut_standard.cpp neut_common.cpp $(UTIL_SOURCE)
	$(CUDA_NVCC) -x cu $(ARCH_DEF) $(FLAGS) -o neut_std neut_standard.cpp --restrict -include $(UTIL_PATH)



neut_evt:      Makefile $(HARMONIZE_PATH) neut_harmonized.cpp neut_common.cpp $(UTIL_SOURCE)
	$(CUDA_NVCC) -x cu -DCACHE $(ARCH_DEF) $(FLAGS) -o neut_evt neut_harmonized.cpp -DEVENT --restrict -include $(HARMONIZE_PATH)

neut_evt_level:      Makefile $(HARMONIZE_PATH) neut_harmonized.cpp neut_common.cpp $(UTIL_SOURCE)
	$(CUDA_NVCC) -x cu -DCACHE -DLEVEL_CHECK $(ARCH_DEF) $(FLAGS) -o neut_evt_level neut_harmonized.cpp -DEVENT --restrict -include $(HARMONIZE_PATH)

