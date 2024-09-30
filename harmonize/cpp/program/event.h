


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
	struct Lookup { typedef typename PromiseUnionType::template Lookup<TYPE>::type type; };


	CONST_SWITCH(size_t,GROUP_SIZE,adapt::WARP_SIZE)


	static const size_t       WORK_GROUP_SIZE  = GROUP_SIZE;

	/*
	// A set of halting condition flags
	*/
	static const unsigned int BAD_FUNC_ID_FLAG	= 0x00000001;
	static const unsigned int OUT_OF_MEM_FLAG	= 0x00000002;
	static const unsigned int EARLY_HALT_FLAG	= 0x40000000;
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


	struct Status {
		unsigned int checkout;
		unsigned int flip_count;
		unsigned int flags;
	};

	/*
	// This struct represents the entire set of data structures that must be stored in main
	// memory to track the state of the program defined by the developer as well as the state
	// of the context which is driving execution.
	*/
	struct DeviceContext {

		typedef		ProgramType       ParentProgramType;

		Status        *status;
		unsigned int   load_margin;
		util::iter::IOBuffer<PromiseUnionType,AdrType> *event_io[PromiseUnionType::Info::COUNT];
	};


	/*
	// Instances wrap around their program scope's DeviceContext. These differ from a program's
	// DeviceContext object in that they perform automatic deallocation as soon as they drop
	// out of scope.
	*/
	struct Instance {


		util::host::DevBuf<Status> status;
		util::host::DevObj<util::iter::IOBuffer<PromiseUnionType>> event_io[PromiseUnionType::Info::COUNT];
		DeviceState device_state;

		__host__ Instance (size_t io_size, DeviceState gs)
			: device_state(gs)
		{
			for( unsigned int i=0; i<PromiseUnionType::Info::COUNT; i++){
				event_io[i] = util::host::DevObj<util::iter::IOBuffer<PromiseUnionType>>(io_size);
			}
			status << Status{0u,0u,0u};
		}

		__host__ DeviceContext to_context(){

			DeviceContext result;

			result.status = status;
			for( unsigned int i=0; i<PromiseUnionType::Info::COUNT; i++){
				result.event_io[i] = event_io[i];
			}

			return result;

		}

		__host__ void clear_flags(){
			unsigned int zero = 0;
			util::host::auto_throw(adapt::GPUrtMemcpy(
				&(((Status*)status)->flags),
				&zero,
				sizeof(unsigned int),
				adapt::GPUrtMemcpyHostToDevice
			));
		}

		__host__ Status load_status(){
			Status host_status;
			util::host::auto_throw(adapt::GPUrtMemcpy(
				&host_status,
				status,
				sizeof(Status),
				adapt::GPUrtMemcpyDeviceToHost
			));
			return host_status;
		}

		__host__ void throw_on_error_status() {
			Status host_status = load_status();
			unsigned int flags = host_status.flags;
			if (flags & 0x1) {
				throw std::runtime_error("RUNTIME ERROR: Call made for async function wihtout a valid function ID.");
			} else if (flags & 0x2) {
				throw std::runtime_error("RUNTIME ERROR: Resource allocation failure during execution.");
			}
		}

		__host__ bool complete(){

			throw_on_error_status();

			bool complete = true;
			for( unsigned int i=0; i<PromiseUnionType::Info::COUNT; i++){
				event_io[i].pull_data();
				check_error();
				size_t item_count = event_io[i].host_copy().input_iter.limit;
				if( item_count != 0 ){
					complete = false;
				}

				// printf("\nitem count: %zu\n",item_count);
				/*/
				PromiseUnionType *host_array = new PromiseUnionType[item_count];
				util::host::auto_throw(adapt::GPUrtMemcpy(
					host_array,
					event_io[i].host_copy().input_pointer(),
					sizeof(PromiseUnionType)*item_count,
					adapt::GPUrtMemcpyDeviceToHost
				));
				for(size_t i=0; i<item_count; i++) {
					char * data = (char*) (host_array + i);
					for(size_t j=0; j<sizeof(PromiseUnionType); j++) {
						printf("%02hhx",data[j]);
					}
					printf(",\n");
				}
				printf("------\n");
				delete[] host_array;
				//*/
			}
			return complete;

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

		atomicOr(&_dev_ctx.status->flags,flag_bits);

	}


	/*
	// Unsets the bits in the status_flags field of the stack according to the given flag bits.
	*/
	__device__  void unset_flags(unsigned int flag_bits){

		atomicAnd(&_dev_ctx.status->_flags,~flag_bits);

	}

	__device__ bool any_flags_set() {
		return (atomicAdd(&_dev_ctx.status->flags,0) != 0);
	}


	static void check_error(){

		adapt::GPUrtError_t status = adapt::GPUrtGetLastError();

		if(status != adapt::GPUrtSuccess){
			const char* err_str = adapt::GPUrtGetErrorString(status);
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
		//*/

		// Investigate push_idx

		if( _dev_ctx.event_io[io_index]->push_index(promise_index) ){
			//printf("{%d : -> %d}\n",io_index,promise_index);
			/*/
			printf("\n(");
			char *data = (char*) &param_value;
			for(size_t i=0; i<sizeof(param_value); i++){
				printf("%02hhx",data[i]);
			}
			printf(")\n");
			//*/
			_dev_ctx.event_io[io_index]->output_pointer()[promise_index].template cast<TYPE>() = param_value;
			//printf("\n\nDoing an async call!\n\n");
		} else {
			//printf("\n\nRan out of space!\n\n");
			set_flags(OUT_OF_MEM_FLAG);
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

		__shared__ util::iter::ArrayIter<PromiseUnionType,util::iter::AtomicIter,unsigned int> group_work;
		__shared__ bool done;
		__shared__ bool early_halt;
		__shared__ OpDisc func_id;

		__syncthreads();
		if( threadIdx.x == 0 ){
			early_halt = false;
			group_work.reset(NULL,util::iter::Iter<unsigned int>(0,0));
		}
		__syncthreads();

		/* The execution loop. */
		unsigned int loop_lim = 0xFFFFF;
		unsigned int loop_count = 0;
		while(!early_halt){
			__syncthreads();
			if( threadIdx.x == 0 ) {
				done = true;

				if( any_flags_set() ){
					early_halt = true;
				} else {
					for(unsigned int i=0; i < PromiseUnionType::Info::COUNT; i++){
						if( !_dev_ctx.event_io[i]->input_empty() ){
							done = false;
							func_id = static_cast<OpDisc>(i);
							group_work = _dev_ctx.event_io[i]->pull_span(chunk_size*GROUP_SIZE);
							if (!group_work.done()){
								break;
							}
						}
					}
				}

			}

			__syncthreads();
			if( done ){
				__shared__ bool should_make_work;
				__syncthreads();
				if( threadIdx.x == 0 ) {
					should_make_work = !early_halt;
				}
				__syncthreads();
				while(should_make_work){
					if( threadIdx.x == 0 ) {
						for(int i=0; i<PromiseUnionType::Info::COUNT; i++){
							int load = atomicAdd(&(_dev_ctx.event_io[i]->output_iter.value),0u);
							if(load >= (_dev_ctx.event_io[i]->output_iter.limit / 2) ){
								//printf("(Hit load margin!)\n");
								should_make_work = false;
							}
						}
					}
					if(should_make_work){
						if( threadIdx.x == 0 ) {
							rc_printf("(Making work!)\n");
						}
						should_make_work = PROGRAM_SPEC::make_work(*this);
						if( (threadIdx.x == 0) && !should_make_work) {
							rc_printf("(No more work!)\n");
						}
					}
				}
				break;
			} else {
				util::iter::ArrayIter<PromiseUnionType,util::iter::Iter,unsigned int> thread_work;
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
			__threadfence();
			unsigned int checkout_index = atomicAdd(&_dev_ctx.status->checkout,1);
			if( checkout_index == (gridDim.x - 1) ){
				__threadfence();
				atomicExch(&_dev_ctx.status->checkout,0);
				__threadfence();
				for(unsigned int i=0; i < PromiseUnionType::Info::COUNT; i++){
					_dev_ctx.event_io[i]->flip();
				}
				atomicAdd(&_dev_ctx.status->flip_count,1u);
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

	__device__ void halt_early() {
		set_flags(EARLY_HALT_FLAG);
	}

};



#undef MEMBER_SWITCH
#undef CONST_SWITCH


#endif


