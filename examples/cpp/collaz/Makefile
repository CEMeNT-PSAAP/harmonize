
# Use this on lassen
# CUDA_NVCC	= /usr/tce/packages/cuda/cuda-11.5.0/bin/nvcc

CUDA_NVCC       = nvcc
HARMONIZE_PATH  = ../../../harmonize/cpp/harmonize.h

all: clz_hrm.exe clz_std.exe

clean:
	rm -f clz_hrm.exe
	rm -f clz_std.exe

clz_hrm.exe: collaz_hrm.cpp $(HARMONIZE_PATH) $(UTIL_SOURCE)
	$(CUDA_NVCC) -x cu -std=c++11 --restrict -o clz_hrm.exe collaz_hrm.cpp -include $(HARMONIZE_PATH)

clz_std.exe: collaz_std.cpp $(HARMONIZE_PATH)
	$(CUDA_NVCC) -x cu -std=c++11 -o clz_std.exe collaz_std.cpp -include $(HARMONIZE_PATH)
