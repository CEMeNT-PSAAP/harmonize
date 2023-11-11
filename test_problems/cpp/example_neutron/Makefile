
#use if on DGX or other OSU GPU systems
#CUDA_NVCC	=	/usr/local/apps/cuda/cuda-10.1/bin/nvcc

#otherwise, use this
CUDA_NVCC	=	nvcc
HIPCC	    = 	hipcc

FLAGS		=	-std=c++11 --compiler-options -fdiagnostics-color=always -g

HARMONIZE_PATH  =       ../../../harmonize.h
UTIL_PATH       =       ../../../util/util.h
UTIL_SOURCE     =       $(shell find ../../../util -type f -name '*.h' )


ifndef ARCH
ARCH_DEF=
else
ARCH_DEF=-arch=$(ARCH)
endif


all: neut


clean:
	rm -f neut

neut:      Makefile $(HARMONIZE_PATH) program.cpp logic.cpp $(UTIL_SOURCE)
	$(CUDA_NVCC) -x cu $(ARCH_DEF) $(FLAGS) -o neut program.cpp --restrict -include $(HARMONIZE_PATH)



neut_evt_amd: Makefile $(HARMONIZE_PATH) program.cpp logic.cpp $(UTIL_SOURCE)
	hipify-clang program.cpp     --default-preprocessor -- -include $(HARMONIZE_PATH) --std=c++17 -DEVENT_MODE
	hipify-clang logic.cpp       --default-preprocessor -- -include $(HARMONIZE_PATH) --std=c++17 -DEVENT_MODE
	$(HIPCC) -x hip -std=c++11 -o neut_evt_amd program.cpp.hip -include $(HARMONIZE_PATH) --std=c++17 -DEVENT_MODE -g
