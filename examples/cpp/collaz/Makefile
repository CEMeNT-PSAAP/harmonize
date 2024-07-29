# Use this on lassen
# CC_PATH	= /usr/tce/packages/cuda/cuda-11.5.0/bin/nvcc

TARGET          = CUDA
HARMONIZE_PATH  = ../../../harmonize/cpp/harmonize.h

ifeq ($(TARGET), CUDA)
CC_PATH ?= nvcc
OPTS    ?= --restrict -x cu
endif

ifeq ($(TARGET), HIP)
CC_PATH ?= hipcc
OPTS    ?= -x hip
endif

CC_PATH ?= nvcc
OPTS    ?= --restrict


all: clz_hrm.exe clz_std.exe

clean:
	rm -f clz_hrm.exe
	rm -f clz_std.exe

clz_hrm.exe: collaz_hrm.cpp $(HARMONIZE_PATH) $(UTIL_SOURCE) Makefile
	$(CC_PATH) -std=c++17 $(OPTS) $(ARCH) -o clz_hrm.exe collaz_hrm.cpp -include $(HARMONIZE_PATH) -g

clz_std.exe: collaz_std.cpp $(HARMONIZE_PATH) Makefile
	$(CC_PATH) -std=c++17 $(OPTS) $(ARCH) -o clz_std.exe collaz_std.cpp -include $(HARMONIZE_PATH) -g

