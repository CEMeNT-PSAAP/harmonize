


#ifndef HARMONIZE_PROGRAM_EVENT
#define HARMONIZE_PROGRAM_EVENT

#include "../async/mod.h"
#include "../mem/mod.h"

//! This macro is currently unused and partially implemented. When finished, it will
//! operate like the `MEMBER_SWITCH` macro, but for internal templates.
#define TEMPLATE_MEMBER_SWITCH(NAME,DEFAULT_TYPE) \
	template<class PROG,class DEFAULT> static auto NAME##Lookup (int)  -> DEFAULT; \
	template<class PROG,class DEFAULT> static auto NAME##Lookup (bool) -> typename Specializer<typename PROG::NAME>::Type; \
	typedef decltype(NAME##Lookup<PROGRAM_SPEC,DEFAULT_TYPE>(true)) NAME;


//! This macro inserts code that detects whether or not an internal type is defined by
//! the type `PROGRAM_SPEC` (the parameter name used to refer to a program specification)
//! and either defines an internal type that duplicates the internal type of `PROGRAM_SPEC`
//! or defines an internal type based off of a default.
#define MEMBER_SWITCH(NAME,DEFAULT_TYPE) \
	template<class PROG,class DEFAULT> static auto NAME##Lookup (int)  -> DEFAULT; \
	template<class PROG,class DEFAULT> static auto NAME##Lookup (bool) -> typename PROG::NAME; \
	typedef decltype(NAME##Lookup<PROGRAM_SPEC,DEFAULT_TYPE>(true)) NAME;

//! This macro behaves the same as `MEMBER_SWITCH`, but for internal static constants,
//! rather than for types
#define CONST_SWITCH(TYPE,NAME,DEFAULT_VAL) \
	template<class PROG,TYPE DEFAULT> static auto NAME##Lookup (int)  -> std::integral_constant<TYPE,    DEFAULT>; \
	template<class PROG,TYPE DEFAULT> static auto NAME##Lookup (bool) -> std::integral_constant<TYPE, PROG::NAME>; \
	static const size_t NAME = decltype(NAME##Lookup<PROGRAM_SPEC,DEFAULT_VAL>(true))::value;




template<typename PROGRAM_SPEC>
class EventProgram
{


	public:


	typedef EventProgram<PROGRAM_SPEC> ProgramType;


	template<class TYPE,class DEFAULT>
	static auto ThreadStateLookup (int)   -> DEFAULT;

	template<class TYPE,class DEFAULT>
	static auto ThreadStateLookup (double) -> typename TYPE::ThreadState;
	/*
	struct AdrTypeDefault { typedef unsigned int AdrType; };
	template<class TYPE> using  AdrTypeCheck = typename TYPE::AdrType;
	typedef typename type_or<AdrTypeDefault,AdrTypeCheck,PROGRAM_SPEC>::AdrType AdrType;
	*/

	MEMBER_SWITCH(    AdrType,unsigned int)
	MEMBER_SWITCH(      OpSet,   OpUnion<>)
	MEMBER_SWITCH(DeviceState,   VoidState)
	MEMBER_SWITCH( GroupState,   VoidState)

	typedef decltype(ThreadStateLookup<PROGRAM_SPEC,VoidState>(1.0)) ThreadState;

	typedef PromiseUnion<OpSet> PromiseUnionType;

	template<typename TYPE>
	struct Lookup { typedef typename PromiseUnionType::Lookup<TYPE>::type type; };


	CONST_SWITCH(size_t,GROUP_SIZE,32)


	static const size_t       WORK_GROUP_SIZE  = GROUP_SIZE;

	/*
	// A set of halting condition flags
	*/
	static const unsigned int BAD_FUNC_ID_FLAG	= 0x00000001;
	static const unsigned int COMPLETION_FLAG	= 0x80000000;


	/*
	// The number of async functions present in the program.
	*/
	static const unsigned char FN_ID_COUNT = PromiseUnionType::Info::COUNT;

	/*
	// This struct represents the entire set of data structures that must be stored in thread
	// memory to track te state of the program defined by the developer as well as the state of
	// the context which is driving exection.
	*/
	struct ThreadContext {

		unsigned int	thread_id;
		unsigned int	rand_state;

	};



	/*
	// This struct represents the entire set of data structures that must be stored in group
	// memory to track te state of the program defined by the developer as well as the state of
	// the context which is driving exection.
	*/
	struct GroupContext {

	};


	/*
	// This struct represents the entire set of data structures that must be stored in main
	// memory to track the state of the program defined by the developer as well as the state
	// of the context which is driving execution.
	*/
	struct DeviceContext {

		typedef		ProgramType       ParentProgramType;

		unsigned int  *checkout;
		unsigned int   load_margin;
		util::iter::IOBuffer<PromiseUnionType,AdrType> *event_io[PromiseUnionType::Info::COUNT];
	};


	/*
	// Instances wrap around their program scope's DeviceContext. These differ from a program's
	// DeviceContext object in that they perform automatic deallocation as soon as they drop
	// out of scope.
	*/
	struct Instance {


		util::host::DevBuf<unsigned int> checkout;
		util::host::DevObj<util::iter::IOBuffer<PromiseUnionType>> event_io[PromiseUnionType::Info::COUNT];
		DeviceState device_state;

		__host__ Instance (size_t io_size, DeviceState gs)
			: device_state(gs)
		{
			for( unsigned int i=0; i<PromiseUnionType::Info::COUNT; i++){
				event_io[i] = util::host::DevObj<util::iter::IOBuffer<PromiseUnionType>>(io_size);
			}
			checkout<< 0u;
		}

		__host__ DeviceContext to_context(){

			DeviceContext result;

			result.checkout = checkout;
			for( unsigned int i=0; i<PromiseUnionType::Info::COUNT; i++){
				result.event_io[i] = event_io[i];
			}

			return result;

		}

		__host__ bool complete(){

			for( unsigned int i=0; i<PromiseUnionType::Info::COUNT; i++){
				event_io[i].pull_data();
				check_error();
				if( ! event_io[i].host_copy().input_iter.limit == 0 ){
					return false;
				}
			}
			return true;

		}

	};


	protected:
	public:

	DeviceContext & _dev_ctx;
	GroupContext  & _grp_ctx;
	ThreadContext & _thd_ctx;


	DeviceState   &   device;
	GroupState    &    group;
	ThreadState   &   thread;


	__device__
	EventProgram<PROGRAM_SPEC>
	(
		DeviceContext & d_c,
		GroupContext  & g_c,
		ThreadContext & t_c,

		DeviceState   &    d,
		GroupState    &    g,
		ThreadState   &    t
	)
		: _dev_ctx(d_c)
		, _grp_ctx(g_c)
		, _thd_ctx(t_c)
		, device  (d)
		, group   (g)
		, thread  (t)
	{}



	protected:

	/*
	// Initializes the shared state of a work group, which is stored as a ctx_shared struct. This
	// is mainly done by initializing handles to the arena, pool, and stack, setting the current
	// level to null, setting the stash iterator to null, and zeroing the stash.
	*/
	__device__  void init_group(){ }

	/*
	// Initializes the local state of a thread, which is just the device id of the thread and the
	// state used by the thread to generate random numbers for stochastic choices needed to manage
	// the runtime state.
	*/
	__device__ void init_thread(){

		_thd_ctx.thread_id  = (blockIdx.x * blockDim.x) + threadIdx.x;
		_thd_ctx.rand_state = _thd_ctx.thread_id;

	}


	/*
	// Sets the bits in the status_flags field of the stack according to the given flag bits.
	*/
	 __device__  void set_flags(unsigned int flag_bits){

		atomicOr(&_dev_ctx.stack->status_flags,flag_bits);

	}


	/*
	// Unsets the bits in the status_flags field of the stack according to the given flag bits.
	*/
	 __device__  void unset_flags(unsigned int flag_bits){

		atomicAnd(&_dev_ctx.stack->status_flags,~flag_bits);

	}


	 static void check_error(){

		cudaError_t status = cudaGetLastError();

		if(status != cudaSuccess){
			const char* err_str = cudaGetErrorString(status);
			printf("ERROR: \"%s\"\n",err_str);
		}

	}



	template<typename TYPE>
	__device__  void async_call_cast(int depth_delta, Promise<TYPE> param_value){
		AdrType promise_index = 0;
		AdrType io_index = static_cast<AdrType>(Lookup<TYPE>::type::DISC);
		/*
		printf("Event io at index %d is at %p with buffers at %p and %p\n",
			io_index,
			_dev_ctx.event_io[io_index],
			_dev_ctx.event_io[io_index]->data_a,
			_dev_ctx.event_io[io_index]->data_b
		);
		*/
		if( _dev_ctx.event_io[io_index]->push_idx(promise_index) ){
			_dev_ctx.event_io[io_index]->output_ptr()[promise_index].template cast<TYPE>() = param_value;
		}
	}


	template<typename TYPE>
	__device__  void immediate_call_cast(Promise<TYPE> param_value){
		PromiseUnionType promise;
		promise.template cast<TYPE>() = param_value;
		promise.template rigid_eval<ProgramType,TYPE>(*this);
		//promise_eval<ProgramType,FUNC_ID>(param_value);
	}


	public:

	__device__ void init_program() {

	}

	/*
	// The workhorse of the program. This function executes until either a halting condition
	// is encountered or a maximum number of processing cycles has occured. This makes sure
	// that long-running programs don't time out on the GPU. In practice, cycle_count may have
	// to be tuned to the average cycle execution time for a given application. This could
	// potentially be automated using an exponential backoff heuristic.
	*/
	 __device__  void exec(unsigned int chunk_size){

		/* Initialize per-warp resources */
		init_group();

		/* Initialize per-thread resources */
		init_thread();

		PROGRAM_SPEC::initialize(*this);

		__shared__ util::iter::GroupArrayIter<PromiseUnionType,unsigned int> group_work;
		__shared__ bool done;
		__shared__ OpDisc func_id;

		util::iter::GroupIter<unsigned int> the_iter;
		the_iter.reset(0,0);
		__syncthreads();
		if( util::current_leader() ){
			group_work = util::iter::GroupArrayIter<PromiseUnionType,unsigned int> (NULL,the_iter);
		}
		__syncthreads();


		/* The execution loop. */
		unsigned int loop_lim = 0xFFFFF;
		unsigned int loop_count = 0;
		while(true){
			__syncthreads();
			if( util::current_leader() ) {
				done = true;
				for(unsigned int i=0; i < PromiseUnionType::Info::COUNT; i++){
					if( !_dev_ctx.event_io[i]->input_empty() ){
						done = false;
						func_id = static_cast<OpDisc>(i);
						group_work = _dev_ctx.event_io[i]->pull_group_span(chunk_size*GROUP_SIZE);
						break;
					}
				}
			}
			__syncthreads();
			if( done ){
				__shared__ bool should_make_work;
				if( util::current_leader() ) {
					should_make_work = true;
				}
				__syncthreads();
				while(should_make_work){
					if( util::current_leader() ) {
						for(int i=0; i<PromiseUnionType::Info::COUNT; i++){
							int load = atomicAdd(&(_dev_ctx.event_io[i]->output_iter.value),0u);
							if(load >= _dev_ctx.load_margin){
								should_make_work = false;
							}
						}
					}
					if(should_make_work){
						should_make_work = PROGRAM_SPEC::make_work(*this);
					}
				}
				break;
			} else {

				util::iter::ArrayIter<PromiseUnionType,unsigned int> thread_work;
				thread_work = group_work.leap(chunk_size);
				PromiseUnionType promise;
				while( thread_work.step_val(promise) ){
					promise.template loose_eval<ProgramType>(*this,func_id);
				}

				if(loop_count < loop_lim){
					loop_count++;
				} else {
					break;
				}
			}

		}


		__syncthreads();

		PROGRAM_SPEC::finalize(*this);

		__threadfence();
		__syncthreads();

		if( threadIdx.x == 0 ){
			unsigned int checkout_index = atomicAdd(_dev_ctx.checkout,1);
			if( checkout_index == (gridDim.x - 1) ){
				atomicExch(_dev_ctx.checkout,0);
				 for(unsigned int i=0; i < PromiseUnionType::Info::COUNT; i++){
					 _dev_ctx.event_io[i]->flip();
				 }

			}
		}



	}

	template<typename TYPE,typename... ARGS>
	__device__  void async(ARGS... args){
		async_call_cast<TYPE>(0,Promise<TYPE>(args...));
	}

	template<typename TYPE>
	__device__  void async_call(Promise<TYPE> promise){
		async_call_cast<TYPE>(0,promise);
	}

	template<typename TYPE,typename... ARGS>
	__device__  void sync(ARGS... args){
		immediate_call_cast<TYPE>(Promise<TYPE>(args...));
	}

	template<typename TYPE>
	__device__  void sync_call(Promise<TYPE> promise){
		immediate_call_cast<TYPE>(0,promise);
	}

	template<typename TYPE>
	__device__ float load_fraction()
	{
		return _dev_ctx.event_io[Lookup<TYPE>::type::DISC]->output_fill_fraction_sync();
	}

};



#undef MEMBER_SWITCH
#undef CONST_SWITCH


#endif


