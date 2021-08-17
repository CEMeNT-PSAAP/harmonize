CUDA_PATH	=	/usr/local/apps/cuda/cuda-10.1
CUDA_BIN_PATH	=	$(CUDA_PATH)/bin
CUDA_NVCC	=	$(CUDA_BIN_PATH)/nvcc
STD		=	-std=c++11

HARMONIZE_PATH  =       ../../../harmonize.cu


ifneq ($(origin WG_COUNT), undefined)
DEF_WG_COUNT=-DDEF_WG_COUNT=$(WG_COUNT)
endif

ifneq ($(origin STD_WG_COUNT), undefined)
DEF_STD_WG_COUNT=-DDEF_WG_COUNT=$(STD_WG_COUNT)
endif


ifneq ($(origin WG_SIZE), undefined)
DEF_WG_SIZE=-DDEF_WG_SIZE=$(WG_SIZE)
endif


ifneq ($(origin FUNC_ID_COUNT), undefined)
DEF_FUNC_ID_COUNT=-DDEF_FUNCTION_ID_COUNT=$(FUNC_ID_COUNT)
endif

ifneq ($(origin THUNK_SIZE), undefined)
DEF_THUNK_SIZE=-DDEF_THUNK_SIZE=$(THUNK_SIZE)
endif

ifneq ($(origin STASH_SIZE), undefined)
DEF_STASH_SIZE=-DDEF_STASH_SIZE=$(STASH_SIZE)
endif

ifneq ($(origin RETRY_LIMIT), undefined)
DEF_RETRY_LIMIT=-DDEF_RETRY_LIMIT=$(RETRY_LIMIT)
endif

ifneq ($(origin QUEUE_WIDTH), undefined)
DEF_QUEUE_WIDTH=-DDEF_QUEUE_WIDTH=$(QUEUE_WIDTH)
endif

ifneq ($(origin ARENA_SIZE), undefined)
DEF_ARENA_SIZE=-DDEF_ARENA_SIZE=$(ARENA_SIZE)
endif

ifneq ($(origin INDIRECT), undefined)
DEF_INDIRECT=-DINDIRECT
DEF_THUNK_SIZE=-DDEF_THUNK_SIZE=1
else
DEF_INDIRECT=
ifeq ($(origin NEU_3D), undefined)
DEF_THUNK_SIZE=-DDEF_THUNK_SIZE=4
else
DEF_THUNK_SIZE=-DDEF_THUNK_SIZE=8
endif
endif

ifeq ($(origin NEU_3D), undefined)
DEF_NEU_3D=
else
DEF_NEU_3D=-DNEUTRON_3D
endif


all: fiss_hrm fiss_std

hard: do_touch all

do_touch:
	touch fiss_harmonized.cu
	touch fiss_standard.cu
	touch fiss_common.cu


fiss_hrm:       $(HARMONIZE_PATH) fiss_harmonized.cu fiss_common.cu
	$(CUDA_NVCC) $(STD) -o fiss_hrm fiss_harmonized.cu --restrict -include $(HARMONIZE_PATH) -DDEFFUNCTION_ID_COUNT=1 -DDEF_STASH_SIZE=4 $(DEF_THUNK_SIZE) $(DEF_WG_COUNT) $(DEF_WG_SIZE) $(DEF_QUEUE_WIDTH) $(DEF_ARENA_SIZE) $(DEF_INDIRECT) $(DEF_NEU_3D)



fiss_std:       fiss_standard.cu fiss_common.cu
	$(CUDA_NVCC) $(STD) -o fiss_std fiss_standard.cu --restrict $(DEF_STD_WG_COUNT) $(DEF_WG_SIZE) $(DEF_NEU_3D)


