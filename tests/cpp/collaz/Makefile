
#use if on DGX or other OSU GPU systems
#CUDA_NVCC	=	/usr/local/apps/cuda/cuda-10.1/bin/nvcc

#otherwise, use this
CUDA_NVCC	=	nvcc


HARMONIZE_PATH  =       ../../../include/harmonize.h

all: clz_hrm.exe clz_std.exe

clean:
	rm -f clz_hrm.exe
	rm -f clz_std.exe

clz_hrm.exe: collaz_hrm.cpp $(HARMONIZE_PATH) $(UTIL_SOURCE)
	$(CUDA_NVCC) -x cu -std=c++11 --restrict -o clz_hrm.exe collaz_hrm.cpp -include $(HARMONIZE_PATH)

clz_std.exe: collaz_std.cpp $(HARMONIZE_PATH)
	$(CUDA_NVCC) -x cu -std=c++11 -o clz_std.exe collaz_std.cpp -include $(HARMONIZE_PATH)
