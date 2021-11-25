CUDA_PATH	=	/usr/local/apps/cuda/cuda-10.1
CUDA_BIN_PATH	=	$(CUDA_PATH)/bin
CUDA_NVCC	=	$(CUDA_BIN_PATH)/nvcc
STD		=	-std=c++11

HARMONIZE_PATH  =       ../../../harmonize.cu
UTIL_PATH       =       ../../../util/util.cu
UTIL_SOURCE     =       $(shell find ../../../util -type f -name '*.cu' )


all: fiss_hrm fiss_std fiss_evt

hard: do_touch all

do_touch:
	touch fiss_harmonized.cu
	touch fiss_event.cu
	touch fiss_standard.cu
	touch fiss_common.cu


fiss_hrm:      Makefile ../../../harmonize.cu fiss_harmonized.cu fiss_common.cu $(UTIL_SOURCE)
	$(CUDA_NVCC) $(STD) -o fiss_hrm fiss_harmonized.cu --restrict -include $(HARMONIZE_PATH)


fiss_std:      Makefile fiss_standard.cu fiss_common.cu $(UTIL_SOURCE)
	$(CUDA_NVCC) $(STD) -o fiss_std fiss_standard.cu -g --restrict -include $(UTIL_PATH)


fiss_evt:      Makefile ../../../harmonize.cu fiss_event.cu fiss_common.cu $(UTIL_SOURCE)
	$(CUDA_NVCC) $(STD) -o fiss_evt fiss_event.cu --restrict -include $(HARMONIZE_PATH)

