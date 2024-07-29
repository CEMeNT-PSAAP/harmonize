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

ifndef ARCH
ARCH_DEF=
else
ARCH_DEF=-arch=$(ARCH)
endif


CC_PATH ?= nvcc
OPTS    ?= --restrict



all: neut.exe

clean:
	rm -f neut.exe

neut.exe: program.cpp logic.cpp $(HARMONIZE_PATH) $(UTIL_SOURCE) Makefile
	$(CC_PATH) -std=c++17 $(OPTS) $(ARCH) -o neut.exe program.cpp -include $(HARMONIZE_PATH) -g


