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





all: collaz_hrm_split collaz_hrm_unified collaz_van

hard: do_touch all

do_touch:
	touch collaz.cu
	touch vanilla_collaz.cu

collaz_hrm_split:       ../../harmonize.cu collaz.cu
	$(CUDA_NVCC) -o collaz_hrm_split collaz.cu -DSPLIT_COLLAZ -DDEFFUNCTION_ID_COUNT=3 -DDEF_STASH_SIZE=5 $(DEF_SM_COUNT) $(DEF_SPAN) $(DEF_QUEUE_WIDTH) $(DEF_ARENA_SIZE)


collaz_hrm_unified:       ../../harmonize.cu collaz.cu
	$(CUDA_NVCC) -o collaz_hrm_unified collaz.cu -DDEFFUNCTION_ID_COUNT=2 -DDEF_STASH_SIZE=4 $(DEF_SM_COUNT) $(DEF_SPAN) $(DEF_QUEUE_WIDTH) $(DEF_ARENA_SIZE)


collaz_van:       vanilla_collaz.cu
	$(CUDA_NVCC) -o collaz_van vanilla_collaz.cu $(DEF_SM_COUNT) $(DEF_SPAN)
