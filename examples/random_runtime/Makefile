CUDA_PATH	=	/usr/local/apps/cuda/cuda-10.1
CUDA_BIN_PATH	=	$(CUDA_PATH)/bin
CUDA_NVCC	=	$(CUDA_BIN_PATH)/nvcc

ifneq ($(origin SM_COUNT), undefined)
DEF_SM_COUNT=-DDEF_MULTIPROCESSOR_COUNT=$(SM_COUNT)
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

ifneq ($(origin SPAN), undefined)
DEF_SPAN=-DDEF_SPAN=$(SPAN)
endif

ifneq ($(origin QUEUE_WIDTH), undefined)
DEF_QUEUE_WIDTH=-DDEF_QUEUE_WIDTH=$(QUEUE_WIDTH)
endif

ifneq ($(origin ARENA_SIZE), undefined)
DEF_ARENA_SIZE=-DDEF_ARENA_SIZE=$(ARENA_SIZE)
endif

ifneq ($(origin THRESHOLD), undefined)
DEF_THRESHOLD=-DDEF_THRESHOLD=$(THRESHOLD)
endif

ifneq ($(origin INNER_ITER), undefined)
DEF_INNER_ITER=-DINNER_ITER_COUNT=$(INNER_ITER)
endif





all: ranrun_hrm ranrun_std

hard: do_touch all

do_touch:
	touch ranrun_harmonized.cu
	touch ranrun_standard.cu


ranrun_hrm:       ../../harmonize.cu ranrun_harmonized.cu
	$(CUDA_NVCC) -o ranrun_hrm ranrun_harmonized.cu -DDEFFUNCTION_ID_COUNT=2 -DDEF_STASH_SIZE=4 -DDEF_THUNK_SIZE=3 $(DEF_SM_COUNT) $(DEF_SPAN) $(DEF_QUEUE_WIDTH) $(DEF_ARENA_SIZE) $(DEF_THRESHOLD) $(DEF_INNER_ITER)


ranrun_std:       ranrun_standard.cu
	$(CUDA_NVCC) -o ranrun_std ranrun_standard.cu $(DEF_SM_COUNT) $(DEF_SPAN) $(DEF_THRESHOLD)
