

#ifndef HARMONIZE_ASYNC_FUTURE
#define HARMONIZE_ASYNC_FUTURE

//! A `Future` is a value tied to a barrier.
template<typename TYPE, typename BARRIER>
struct Future
{
	using BarrierType = BARRIER;
	using Type        = TYPE;

	BarrierType barrier;
	Type        data;

	Future<TYPE,BARRIER>() = default;

	__host__ __device__ Future<TYPE,BARRIER>(unsigned int semaphore_value)
		: barrier(semaphore_value)
	{}

	template<typename PROGRAM, typename OP_TYPE>
	__device__ void await(PROGRAM program, Promise<OP_TYPE> promise) {
		barrier.await(program,promise);
	}


	template<typename PROGRAM>
	__device__ void fulfill(PROGRAM program) {
		barrier.sub_semaphore(program,1);
	}

};



//! The `ReturnOp` operation is used to resolve dependencies between barriers and peform
//! the data transfers required to fill in future values.
struct ReturnOp {

	using Type = void(*)(TaggedSemaphore*,void*,void*,size_t);

	template<typename PROGRAM>
	__device__ void eval(TaggedSemaphore* sem,void* dst, void* src, size_t size) {

		using UFuture = typename PROGRAM::UFuture;
		using RFuture = typename PROGRAM::RFuture;

		memcpy(dst,src,size);
		__threadfence();
		if(sem->tag == 0){
			UFuture* unit_future = sem;
			unit_future ->fulfill();
		} else {
			RFuture* remap_future = sem;
			remap_future->fulfill();
		}
	}

};

#endif


