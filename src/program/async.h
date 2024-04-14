


#ifndef HARMONIZE_PROGRAM_ASYNC
#define HARMONIZE_PROGRAM_ASYNC



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




////////////////////////////////////////////////////////////////////////////////////
//                                                                                //
// To have a well-defined Harmonize program, we need to define:                   //
//                                                                                //
//   - the set of valid async function identifiers (in the PROMISE_UNION)         //
//   - the number of threads within a work group (and hence the number of         //
//     promises per link). Only a work group size of 32 is currently supported.   //
//   - the type used to index links in their resident arena                       //
//   - the number of queues present in pools and stack frames, respectively       //
//   - the height of the stack (currently, only a flat stack is supported)        //
//   - the states that should be tracked per-group and per-thread                 //
//                                                                                //
////////////////////////////////////////////////////////////////////////////////////
//                                                                                //
//                          .-->(QUEUES_PER_POOL)  .--> (PROMISE_UNION)           //
// (GROUP/THREAD_STATE)    /                      /                               //
//      A   .-----------. /          .--> LINK ----> (GROUP_SIZE)                 //
//      |  /             X          /           \                                 //
//      | /             / `-> ARENA------------. \                                //
// PROG ------> POOL --------------.            \ \                               //
//      | \                         \            \ \                              //
//      |  `--> STACK ----> FRAME -----> QUEUE ------->  ADR --> (ADR_TYPE)       //
//      |             \          \                                                //
//      |              \          `--> (QUEUE_PER_FRAME)                          //
//      V               \                                                         //
// (STASH_SIZE)          `--> (STACK_SIZE)                                        //
//                                                                                //
////////////////////////////////////////////////////////////////////////////////////


//! Switches whether or not lazy allocation is enabled for the allocation of a `WorkLink`
#define LAZY_LINK




//! The class that defines an asynchronous program and all of its types.
template< typename PROGRAM_SPEC >
class AsyncProgram
{


	public:

	typedef AsyncProgram<PROGRAM_SPEC> ProgramType;

	//! Templates for use by `TEMPLATE_MEMBER_SWITCH`
	template<typename BASE>
	struct Specializer
	{
		using Type = BASE;
	};

	template<template<typename>typename BASE>
	struct Specializer<BASE<ProgramType>>
	{
		using Type = BASE<ProgramType>;
	};

	//! Define the type used to address work links
	MEMBER_SWITCH(    AdrType,unsigned int)
	//! Define the set of operations
	MEMBER_SWITCH(      OpSet,   OpUnion<>)

	template<typename OP_SET, typename ADR_TYPE>
	friend class RemappingBarrier;

	template<typename... TYPES>
	friend class ArgTuple;

	//! Define the states stored in global, shared, and private memory
	MEMBER_SWITCH(DeviceState,   VoidState)
	MEMBER_SWITCH( GroupState,   VoidState)
	MEMBER_SWITCH(ThreadState,   VoidState)


	//! Define the type of `PromiseUnion` used by the program.
	typedef PromiseUnion<OpSet> PromiseUnionType;


	//! Used to look up information about the primary `PromiseUnion` type used
	template<typename TYPE>
	struct Lookup { typedef typename PromiseUnionType::Lookup<TYPE>::type type; };

	//! Define internal constants based off of the program specification, or
	//! fall back onto defaults.
	CONST_SWITCH(size_t,STASH_SIZE,16)
	CONST_SWITCH(size_t,FRAME_SIZE,32)
	CONST_SWITCH(size_t, POOL_SIZE,32)
	CONST_SWITCH(size_t,STACK_SIZE, 0)
	CONST_SWITCH(size_t,GROUP_SIZE,32)


	//! Constants used to determine when to spill or fill the stash, and
	//! by how much
	static const size_t        STASH_MARGIN     = 2;
	static const size_t        STASH_HIGH_WATER = STASH_SIZE-STASH_MARGIN;

	//! The number of async functions present in the program.
	static const unsigned char FN_ID_COUNT = PromiseUnionType::Info::COUNT;


	//! During system verification/debugging, this will be used as a cutoff to prevent infinite
	//! looping
	static const unsigned int PUSH_QUEUE_RETRY_LIMIT         = 32;
	static const unsigned int FILL_STASH_RETRY_LIMIT         =  1;
	static const unsigned int FILL_STASH_LINKS_RETRY_LIMIT   = 32;

	static const size_t       WORK_GROUP_SIZE  = GROUP_SIZE;

	//! A set of halting condition flags
	static const unsigned int BAD_FUNC_ID_FLAG	= 0x00000001;
	static const unsigned int STASH_FAIL_FLAG	= 0x00000002;
	static const unsigned int COMPLETION_FLAG	= 0x80000000;
	static const unsigned int EARLY_HALT_FLAG	= 0x40000000;

	//! Defining a set of internal short-hand names for the specializaions used by the class
	typedef util::mem::Adr       <AdrType>             LinkAdrType;
	typedef util::mem::PoolQueue <LinkAdrType>         QueueType;
	typedef WorkFrame       <QueueType,FRAME_SIZE>     FrameType;
	typedef WorkStack       <FrameType,STACK_SIZE>     StackType;
	typedef WorkPool        <QueueType,POOL_SIZE>      PoolType;

	typedef WorkLink        <OpSet, LinkAdrType, WORK_GROUP_SIZE> LinkType;

	typedef WorkArena       <LinkAdrType,LinkType>     ArenaType;



	//! The depth of the partial table (1 if stack is flat, 3 otherwise).
	static const unsigned char PART_TABLE_DEPTH = StackType::PART_MULT;
	static const unsigned char PART_ENTRY_COUNT = FN_ID_COUNT*PART_TABLE_DEPTH;


	static const AdrType       SPARE_LINK_COUNT = 2u;


	//! This struct represents the entire set of data structures that must be stored in thread
	//! memory to track te state of the program defined by the developer as well as the state of
	//! the context which is driving exection.
	struct ThreadContext {

		unsigned int thread_id;
		unsigned int rand_state;
		unsigned int spare_index;
		LinkAdrType  spare_links[SPARE_LINK_COUNT];

	};


	//! A non-atomic promise coalescing structure used to track information about full and
	//! partial work links.
	struct RemapQueue {
		unsigned char count;
		unsigned char full_head;
		unsigned char partial_map[PART_ENTRY_COUNT];
	};


	//! This struct represents the entire set of data structures that must be stored in group
	//! memory to track te state of the program defined by the developer as well as the state of
	//! the context which is driving exection.
	struct GroupContext {

		size_t				level;		// Current level being run

		bool				keep_running;
		bool				busy;
		bool				can_make_work;
		bool				scarce_work;

		unsigned char			exec_head;	// Indexes the link that is/will be evaluated next
		unsigned char			empty_head;	// Head of the linked list of empty links

		RemapQueue			main_queue;

		#ifdef ASYNC_LOADS
		RemapQueue			load_queue;
		cuda::barrier<cuda::thread_scope_system> load_barrier;
		#endif

		unsigned char			link_stash_count; // Number of device-space links stored

		LinkType			stash[STASH_SIZE];
		LinkAdrType			link_stash[STASH_SIZE];


		int				SM_promise_delta;
		unsigned long long int		work_iterator;

		#ifdef BARRIER_SPILL
		RemappingBarrier<OpSet>		spill_barrier;
		#endif

		#ifdef HRM_TIME
		unsigned long long int		time_totals[HRM_TIME];
		#endif


	};


	//! This struct represents the entire set of data structures that must be stored in device
	//! memory to track the state of the program defined by the developer as well as the state
	//! of the context which is driving execution.
	struct DeviceContext {

		typedef		ProgramType	ParentProgramType;

		#ifdef LAZY_LINK
		AdrType*        claim_count;
		#endif

		#ifdef HRM_TIME
		unsigned long long int* time_totals;
		#endif

		ArenaType	arena;
		PoolType*	pool;
		StackType*	stack;

	};


	protected:
	public:

	DeviceContext & _dev_ctx;
	GroupContext  & _grp_ctx;
	ThreadContext & _thd_ctx;


	DeviceState   &   device;
	GroupState    &    group;
	ThreadState   &   thread;

	__device__ AsyncProgram<PROGRAM_SPEC> (
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






	//! Instances wrap around their program scope's DeviceContext. These differ from a program's
	//! DeviceContext object in that they perform automatic deallocation as soon as they drop
	//! out of scope.
	struct Instance {


		AdrType                          arena_size;
		#ifdef LAZY_LINK
		util::host::DevBuf<AdrType>      claim_count;
		#endif
		#ifdef HRM_TIME
		util::host::DevBuf<unsigned long long int> time_totals;
		#endif
		util::host::DevBuf<LinkType>     arena;
		util::host::DevBuf<PoolType>     pool;
		util::host::DevBuf<StackType>    stack;
		DeviceState device_state;

		__host__ Instance (AdrType arsize, DeviceState gs)
			: arena(arsize)
			, pool (1)
			, stack(1)
			, arena_size(arsize)
			, device_state(gs)
			#ifdef LAZY_LINK
			, claim_count(0u)
			#endif
			#ifdef HRM_TIME
			, time_totals((size_t)HRM_TIME)
			#endif
		{
			#ifdef HRM_TIME
			cudaMemset( time_totals, 0, sizeof(unsigned long long int) * HRM_TIME );
			#endif
		}

		__host__ DeviceContext to_context(){

			DeviceContext result;

			#ifdef LAZY_LINK
			result.claim_count  = claim_count;
			#endif
			#ifdef HRM_TIME
			result.time_totals  = time_totals;
			#endif

			result.arena.size   = arena_size;
			result.arena.links  = arena;
			result.pool         = pool ;
			result.stack        = stack;

			return result;

		}

		#ifdef HRM_TIME
		__host__ void print_times(){
			std::vector<unsigned long long int> times;
			time_totals >> times;
			double total = times[0];
			for(unsigned int i=0; i<times.size(); i++){
				double the_time = times[i];
				double prop = 100.0 * (the_time / total);
				printf("T%d: %llu (~%f%)\n",i,times[i],prop );
			}
		}
		#endif

		__host__ bool complete(){

			unsigned int* base_cr_ptr = &(((StackType*)stack)->status_flags);
			unsigned int  base_cr = 0;
			cudaMemcpy(&base_cr,base_cr_ptr,sizeof(unsigned int),cudaMemcpyDeviceToHost);
			check_error();
			return (base_cr == 0);
		}

	};





	protected:

	//! Returns true only if the stash cannot support the allocation of any more work links
	__device__ bool stash_overfilled(){
		#ifdef ASYNC_LOADS
		return (_grp_ctx.main_queue.count+_grp_ctx.load_queue.count) >= STASH_HIGH_WATER;
		#else
		return _grp_ctx.main_queue.count >= STASH_HIGH_WATER;
		#endif
	}

	//! Returns an index into the partial map of a group based off of a function id and a depth. If
	//! an invalid depth or function id is used, PART_ENTRY_COUNT is returned.
	 __device__  unsigned int partial_map_index(
		OpDisc     func_id,
		unsigned int depth,
		unsigned int current_level
	){

		unsigned int the_id = static_cast<unsigned int>(func_id);


		if( the_id >= FN_ID_COUNT){
			return PART_ENTRY_COUNT;
		}

		unsigned int result = the_id;

		if( ! StackType::FLAT ){
			result *= PART_TABLE_DEPTH;
			if( depth == current_level ){
				result += 1;
			} else if ( depth == (current_level+1) ){
				result += 2;
			} else if ( depth != (current_level-1) ){
				result = PART_ENTRY_COUNT;
			}
		}

		return result;

	}


	//! Initializes the shared state of a work group, which is stored as a ctx_shared struct. This
	//! is mainly done by initializing handles to the arena, pool, and stack, setting the current
	//! level to null, setting the stash iterator to null, and zeroing the stash.
	 __device__  void init_group(){

		//printf("Initing group\n");
		unsigned int active = __activemask();

		__syncwarp(active);

		if(util::current_leader()){

			if( StackType::FLAT ){
				_grp_ctx.level = 0;
			} else {
				_grp_ctx.level = StackType::NULL_LEVEL;
			}

			_grp_ctx.main_queue.count = 0;
			_grp_ctx.link_stash_count = 0;
			_grp_ctx.keep_running = true;
			_grp_ctx.busy 	 = false;
			_grp_ctx.can_make_work= true;
			_grp_ctx.exec_head    = STASH_SIZE;
			_grp_ctx.main_queue.full_head    = STASH_SIZE;
			_grp_ctx.empty_head   = 0;
			_grp_ctx.work_iterator= 0;
			_grp_ctx.scarce_work  = false;

			#ifdef BARRIER_SPILL
			_grp_ctx.spill_barrier = RemappingBarrier<OpSet>::blank(1);
			#endif

			for(unsigned int i=0; i<STASH_SIZE; i++){
				_grp_ctx.stash[i].empty(i+1);
			}

			for(unsigned int i=0; i<PART_ENTRY_COUNT; i++){
				_grp_ctx.main_queue.partial_map[i] = STASH_SIZE;
			}

			#ifdef ASYNC_LOADS
			_grp_ctx.load_queue.count = 0;
			_grp_ctx.load_queue.full_head = STASH_SIZE;
			for(unsigned int i=0; i<PART_ENTRY_COUNT; i++){
				_grp_ctx.load_queue.partial_map[i] = STASH_SIZE;
			}
			init(&_grp_ctx.load_barrier,WORK_GROUP_SIZE);
			#endif

			_grp_ctx.SM_promise_delta = 0;

			#ifdef HRM_TIME
			for(unsigned int i=0; i<HRM_TIME; i++){
				_grp_ctx.time_totals[i] = 0;
			}
			beg_time(0);
			#endif

		}

		__syncwarp(active);

	}

	//! Initializes the local state of a thread, which is just the device id of the thread and the
	//! state used by the thread to generate random numbers for stochastic choices needed to manage
	//! the runtime state.
	 __device__ void init_thread(){

		//printf("Initing thread %d\n",threadIdx.x);
		_thd_ctx.thread_id   = (blockIdx.x * blockDim.x) + threadIdx.x;
		_thd_ctx.rand_state  = _thd_ctx.thread_id;
		_thd_ctx.spare_index = 0;
		for(size_t i=0; i<SPARE_LINK_COUNT; i++){
			_thd_ctx.spare_links[i] = LinkAdrType::null;
		}

	}


	//! Sets the bits in the status_flags field of the stack according to the given flag bits.
	 __device__  void set_flags(unsigned int flag_bits){

		atomicOr(&_dev_ctx.stack->status_flags,flag_bits);

	}


	//! Unsets the bits in the status_flags field of the stack according to the given flag bits.
	 __device__  void unset_flags(unsigned int flag_bits){

		atomicAnd(&_dev_ctx.stack->status_flags,~flag_bits);

	}

	//! Returns the current highest level in the stack. Given that this program is highly parallel,
	//! this number inherently cannot be trusted. By the time the value is fetched, the stack could
	//! have a different height or the thread that set the height may not have deposited links in the
	//! corresponding level yet.
	 __device__  unsigned int highest_level(){

		return left_half(_dev_ctx.stack->depth_live);

	}


	//! Returns a reference to the frame at the requested level in the stack.
	 __device__  FrameType& get_frame(unsigned int level){

		return _dev_ctx.stack->frames[level];

	}


	//! Joins two queues such that the right queue is now at the end of the left queue.
	//!
	//! WARNING: NOT THREAD SAFE. Only use this on queues that have been claimed from the stack
	//! atomically. If not, one of the queues manipulated with this function will almost certainly
	//! become malformed at some point.
	 __device__  QueueType join_queues(QueueType left_queue, QueueType right_queue){

		QueueType result;

		/*
		// If either input queue is null, we can simply return the other queue.
		*/
		if( left_queue.is_null() ){
			result = right_queue;
		} else if ( right_queue.is_null() ){
			result = left_queue;
		} else {

			LinkAdrType left_tail_adr  = left_queue .get_tail();
			LinkAdrType right_head_adr = right_queue.get_head();
			LinkAdrType right_tail_adr = right_queue.get_tail();

			//! Find last link in the queue referenced by left_queue.
			LinkType& left_tail = _dev_ctx.arena[left_tail_adr];

			//! Set the index for the tail's successor to the head of the queue referenced by
			//! right_queue.
			left_tail.next = right_head_adr;

			//! Set the right half of the left_queue handle to index the new tail.
			left_queue.set_tail(right_tail_adr);

			result = left_queue;

		}

		return result;

	}


	//! Takes the first link off of the queue and returns the index of the link in the arena. If the
	//! queue is empty, a null address is returned instead.
	//!
	//! WARNING: NOT THREAD SAFE. Only use this on queues that have been claimed from the stack
	//! atomically. If not, one of the queues manipulated with this function will almost certainly
	//! become malformed at some point.
	 __device__  LinkAdrType pop_front(QueueType& queue){

		LinkAdrType result;

		//! Don't try unless the queue is non-null
		if( queue.is_null() ){
			result.adr = LinkAdrType::null;
		} else {
			result = queue.get_head();
			LinkAdrType next = _dev_ctx.arena[result].next;
			queue.set_head(next);
			if(next.adr == LinkAdrType::null){
				queue.set_tail(next);
			} else if ( queue.get_tail() == result ){
				//printf("ERROR: Final link does not have a null next.\n");
				queue.pair.data = QueueType::null;
				return result;
			}
		}
		return result;

	}


	//! Adds the given link to the end of the given queue. This link can NOT be part of another queue,
	//! and its next pointer will be automatically nulled before it is appended. If you need to merge
	//! two queues together, use join_queues.
	//!
	//! WARNING: NOT THREAD SAFE. Only use this on queues that have been claimed from the stack
	//! atomically. If not, one of the queues manipulated with this function will almost certainly
	//! become malformed at some point.
	 __device__  void push_back(QueueType& queue, LinkAdrType link_adr){

		_dev_ctx.arena[link_adr].next = LinkAdrType::null;
		if( queue.is_null() ){
			queue = QueueType(link_adr,link_adr);
		} else {
			LinkAdrType tail = queue.get_tail();
			_dev_ctx.arena[tail].next = link_adr;
			//atomicExch( &(_dev_ctx.arena[tail].next.adr),link_adr.adr);
			queue.set_tail(link_adr);
		}

	}



	//! Attempts to pull a queue from a range of queue slots, trying each slot starting from the given
	//! starting index onto the end of the range and then looping back from the beginning. If, after
	//! trying every slot in the range, no non-null queue was obtained, a QueueType::null value is returned.
	 __device__  QueueType pull_queue(QueueType* src, unsigned int start_index, unsigned int range_size, unsigned int& src_index){

		QueueType result;

		__threadfence();
		//! First iterate from the starting index to the end of the queue range, attempting to
		//! claim a non-null queue until either there are no more slots to try, or the atomic
		//! swap successfuly retrieves something.
		for(unsigned int i=start_index; i < range_size; i++){
			if( src[i].pair.data != QueueType::null ) {
				result.pair.data = atomicExch(&(src[i].pair.data),QueueType::null);
				if( ! result.is_null() ){
					src_index = i;
					__threadfence();
					return result;
				}
			}
		}

		//! Continue searching from the beginning of the range to just before the beginning of the
		//! previous scan.
		for(unsigned int i=0; i < start_index; i++){
			if( src[i].pair.data != QueueType::null ) {
				result.pair.data = atomicExch(&(src[i].pair.data),QueueType::null);
				if( ! result.is_null() ){
					src_index = i;
					__threadfence();
					return result;
				}
			}
		}

		q_printf("COULD NOT PULL QUEUE\n");
		//! Return QueueType::null if nothing is found
		result.pair.data = QueueType::null;
		return result;


	}




	//! Repeatedly tries to push a queue to a destination queue slot by atomic exchanges. If a non
	//! null queue is ever returned by the exchange, it attempts to merge with a subsequent exchange.
	//!
	//! If an invalid queue (one half null, the other half non-null) is ever found at the destination,
	//! the value of that invalid queue is returned, the invalid value is replaced into the queue, and
	//! any work gathered from the replacement is assigned to the second argument. Otherwise, the second
	//! argument is set to a null state.
	 __device__  QueueType push_queue(QueueType& dest, QueueType& queue){

		if( queue.is_null() ){
			return queue;
		}
		#ifdef INF_LOOP_SAFE
		while(true)
		#else
		for(int i=0; i<PUSH_QUEUE_RETRY_LIMIT; i++)
		#endif
		{
			__threadfence();
			//printf("%d: Pushing queue (%d,%d)\n",blockIdx.x,queue.get_head().adr,queue.get_tail().adr);

			QueueType swap;
			swap.pair.data = atomicExch(&dest.pair.data,queue.pair.data);
			//! If our swap returns a non-null queue, we are still stuck with a queue that
			//! needs to be offloaded to the stack. In this case, claim the queue from the
			//! slot just swapped with, merge the two, and attempt again to place the queue
			//! back. With this method, swap failures are bounded by the number of pushes to
			//! the queue slot, with at most one failure per push_queue call, but no guarantee
			//! of which attempt from which call will suffer from an incurred failure.
			if( ! swap.is_null() ){
				//printf("%d: We got queue (%d,%d) when trying to push a queue\n",blockIdx.x,swap.get_head().adr,swap.get_tail().adr);
				QueueType other_swap;
				if( swap.get_head().is_null() || swap.get_tail().is_null() ){
					other_swap.pair.data = atomicExch(&dest.pair.data,swap.pair.data);
					queue = other_swap;
					return swap;
				} else {
					other_swap.pair.data = atomicExch(&dest.pair.data,QueueType::null);
					queue = join_queues(other_swap,swap);
					//printf("%d: Merged it to form queue (%d,%d)\n",blockIdx.x,queue.get_head().adr,queue.get_tail().adr);
				}
			} else {
				//printf("%d: Finally pushed (%d,%d)\n",blockIdx.x,queue.get_head().adr,queue.get_tail().adr);
				break;
			}
		}
		return queue;

	}




	//! Claims a link from the link stash. If no link exists in the stash, LinkAdrType::null is returned.
	 __device__  LinkAdrType claim_stash_link(){

		LinkAdrType link = LinkAdrType::null;
		unsigned int count = _grp_ctx.link_stash_count;
		if(count > 0){
			link = _grp_ctx.link_stash[count-1];
			_grp_ctx.link_stash_count = count - 1;
		}
		q_printf("New link stash count: %d\n",_grp_ctx.link_stash_count);
		return link;

	}



	//! Inserts an empty slot into the stash. This should only be called if there is enough space in
	//! the link stash.
	 __device__  void insert_stash_link(LinkAdrType link){

		unsigned int count = _grp_ctx.link_stash_count;
		_grp_ctx.link_stash[count] = link;
		_grp_ctx.link_stash_count = count + 1;
		q_printf("New link stash count: %d\n",_grp_ctx.link_stash_count);

	}




	//! Claims an empty slot from the stash and returns its index. If no empty slot exists in the stash,
	//! then STASH_SIZE is returned.
	 __device__  unsigned int claim_empty_slot(){

		unsigned int slot = _grp_ctx.empty_head;
		if(slot != STASH_SIZE){
			_grp_ctx.empty_head = _grp_ctx.stash[slot].next.adr;
			db_printf("EMPTY: %d << %d\n",slot,_grp_ctx.empty_head);
		}
		return slot;

	}


	//! Inserts an empty slot into the stash. This should only be called if there is enough space in
	//! the link stash.
	 __device__  void insert_empty_slot(unsigned int slot){

		_grp_ctx.stash[slot].next.adr = _grp_ctx.empty_head;
		db_printf("EMPTY: >> %d -> %d\n",slot,_grp_ctx.empty_head);
		_grp_ctx.empty_head = slot;

	}


	//! Claims a full slot from the stash and returns its index. If no empty slot exists in the stash,
	//! then STASH_SIZE is returned.
	 __device__  unsigned int claim_full_slot(){

		unsigned int slot = _grp_ctx.main_queue.full_head;
		if(slot != STASH_SIZE){
			_grp_ctx.main_queue.full_head = _grp_ctx.stash[slot].next.adr;
			db_printf("FULL : %d << %d\n",slot,_grp_ctx.main_queue.full_head);
		}
		return slot;

	}


	//! Inserts a full slot into the stash. This should only be called if there is enough space in
	//! the link stash.
	 __device__  void insert_full_slot(RemapQueue& queue, unsigned int slot){

		_grp_ctx.stash[slot].next.adr = queue.full_head;
		db_printf("FULL : >> %d -> %d\n",slot,queue.full_head);
		queue.full_head = slot;

	}


	 __device__ void dealloc_links(LinkAdrType* src, size_t count) {


		QueueType queue;
		queue.pair.data = QueueType::null;

		//! Connect all links into a queue
		for(unsigned int i=0; i < count; i++){

			push_back(queue,src[i]);

		}

		//! Push out the queue to the pool
		q_printf("Pushing queue (%d,%d) to pool\n",queue.get_head().adr,queue.get_tail().adr);
		unsigned int dest_idx = util::random_uint(_thd_ctx.rand_state) % POOL_SIZE;
		push_queue(_dev_ctx.pool->queues[dest_idx],queue);

	 }


	 //!
	 //! A thread-safe way of allocating links from global memory. This function is used
	 //! by the warp leader to restock the link stash in between processing promises and
	 //! used by individual threads if its current promise needs to use a link while the
	 //! link stash is empty.
	 //!
	 __device__ size_t alloc_links(LinkAdrType* dst, size_t req_count) {

		size_t alloc_count = 0;
		#ifdef LAZY_LINK

		#if 1
		if( (alloc_count < req_count) && ((*_dev_ctx.claim_count) <  _dev_ctx.arena.size) ){
			AdrType claim_offset = atomicAdd(_dev_ctx.claim_count,req_count);
			unsigned int received = 0;
			if( claim_offset <= (_dev_ctx.arena.size - req_count) ){
				received = req_count;
			} else if ( claim_offset >= _dev_ctx.arena.size ) {
				received = 0;
			} else {
				received = _dev_ctx.arena.size - claim_offset;
			}
			for( unsigned int index=0; index < received; index++){
				//insert_stash_link(LinkAdrType(claim_offset+index));
				dst[alloc_count] = LinkAdrType(claim_offset+index);
				alloc_count++;
			}
		}
		#else
		for(int i=_grp_ctx.link_stash_count; i < threashold; i++){
			if((*_dev_ctx.claim_count) <  _dev_ctx.arena.size ){
				AdrType claim_index = atomicAdd(_dev_ctx.claim_count,1);
				if( claim_index < _dev_ctx.arena.size ){
					//_dev_ctx.arena[claim_index].empty(LinkAdrType::null);
					insert_stash_link(LinkAdrType(claim_index));
				}
			} else {
				break;
			}
		}
		#endif

		#endif


		for(int try_itr=0; try_itr < FILL_STASH_LINKS_RETRY_LIMIT; try_itr++){

			if(alloc_count >= req_count){
				break;
			}
			//! Attempt to pull a queue from the pool. This should be very unlikely to fail unless
			//! almost all links have been exhausted or the pool size is disproportionately small
			//! relative to the number of work groups. In the worst case, this should simply not
			//! al_thd_ctxate any links, and the return value shall report this.
			unsigned int src_index = LinkAdrType::null;
			unsigned int start = util::random_uint(_thd_ctx.rand_state)%POOL_SIZE;
			QueueType queue = pull_queue(_dev_ctx.pool->queues,start,POOL_SIZE,src_index);
			q_printf("Pulled queue (%d,%d) from pool %d\n",queue.get_head().adr,queue.get_tail().adr,src_index);

			//! Keep popping links from the queue until the full number of links have been added or
			//! the queue runs out of links.
			for(int i=alloc_count; i < req_count; i++){
				LinkAdrType link = pop_front(queue);
				if( ! link.is_null() ){
					//insert_stash_link(link);
					dst[alloc_count] = link;
					alloc_count++;
					q_printf("Inserted link %d into link stash\n",link.adr);
				} else {
					break;
				}
			}
			push_queue(_dev_ctx.pool->queues[src_index],queue);
			q_printf("Pushed queue (%d,%d) to pool %d\n",queue.get_head().adr,queue.get_tail().adr,src_index);

		}

		return alloc_count;

	 }

	__device__ LinkAdrType alloc_spare_link(){
		if(_thd_ctx.spare_index > 0){
			_thd_ctx.spare_index--;
			LinkAdrType result;
			result = _thd_ctx.spare_links[_thd_ctx.spare_index];
			//printf("Re-used spare alloc (%d)\n",result.adr);
			return result;
		} else {
			LinkAdrType result = LinkAdrType::null;
			alloc_links(&result,1);
			//printf("Fresh spare alloc (%d)\n",result.adr);
			return result;
		}
	}

	__device__ void dump_spare_link(LinkAdrType link_adr){
		if( link_adr.is_null() ){
			return;
		}
		if(_thd_ctx.spare_index < SPARE_LINK_COUNT){
			//printf("Saving link (%d) for later\n",link_adr.adr);
			_thd_ctx.spare_links[_thd_ctx.spare_index] = link_adr;
			_thd_ctx.spare_index++;
			//printf("New spare index of %d is %d\n",threadIdx.x,_thd_ctx.spare_index);
		} else {
			//printf("Spilling spare link (%d)\n",link_adr.adr);
			dealloc_links(&link_adr,1);
		}
	}

	//! Attempts to fill the link stash to the given threshold with links. This should only ever
	//! be called in a single-threaded manner.
	 __device__  void fill_stash_links(unsigned int threashold){

		unsigned int active = __activemask();
		__syncwarp(active);

		if( util::current_leader() ){

			LinkAdrType    *dst = _grp_ctx.link_stash+_grp_ctx.link_stash_count;
			size_t count = 0;
			if( threashold < _grp_ctx.link_stash_count ){
				count = 0;
			} else if ( threashold >= STASH_SIZE ){
				count = STASH_SIZE - _grp_ctx.link_stash_count;
			} else {
				count = threashold - _grp_ctx.link_stash_count;
			}
			size_t        added = alloc_links(dst,count);
			_grp_ctx.link_stash_count += added;

		}
		__syncwarp(active);

	}




	//! If the number of links in the link stash exceeds the given threshold value, this function frees
	//! enough links to bring the number of links down to the threshold. This should only ever be
	//! called in a single_threaded manner.
	 __device__  void spill_stash_links(unsigned int threashold){

		//! Do not even try if no links can be or need to be removed
		if(threashold >= _grp_ctx.link_stash_count){
			q_printf("Nothing to spill...\n");
			return;
		}

		//! Find where in the link stash to begin removing links

		QueueType queue;
		queue.pair.data = QueueType::null;

		//! Connect all links into a queue
		unsigned int spill_count = _grp_ctx.link_stash_count - threashold;
		for(unsigned int i=0; i < spill_count; i++){

			LinkAdrType link = claim_stash_link();
			q_printf("Claimed link %d from link stash\n",link.adr);
			push_back(queue,link);

		}

		_grp_ctx.link_stash_count = threashold;


		//! Push out the queue to the pool
		q_printf("Pushing queue (%d,%d) to pool\n",queue.get_head().adr,queue.get_tail().adr);
		unsigned int dest_idx = util::random_uint(_thd_ctx.rand_state) % POOL_SIZE;
		push_queue(_dev_ctx.pool->queues[dest_idx],queue);

		q_printf("Pushed queue (%d,%d) to pool\n",queue.get_head().adr,queue.get_tail().adr);

	}




	//! Decrements the child and resident counter of each frame corresponding to a call at level
	//! start_level in the stack returning to a continuation at level end_level in the stack. To reduce
	//! overall contention, decrementations are first pooled through a shared atomic operation before
	//! being applied to the stack.
	//
	//! A call without a continuation should use this function with start_level == end_level, which
	//! simply decrements the resident counter at the call's frame.
	 __device__  void pop_frame_counters(unsigned int start_level, unsigned int end_level){


		unsigned int depth_dec = 0;
		PromiseCountPair delta;
		PromiseCountPair result;

		FrameType& frame = _dev_ctx.stack->frames[start_level];

		//! Decrement the residents counter for the start level
		delta = util::active_count();
		if(util::current_leader()){
			result = atomicSub(&frame.children_residents.data,delta);
			if(result == 0u){
				depth_dec += 1;
			}
		}

		//! Decrement the children counter for the remaining levels
		for(int d=(start_level-1); d >= end_level; d--){
			FrameType& frame = _dev_ctx.stack->frames[d];
			delta = util::active_count();
			if(util::current_leader()){
				result = atomicSub(&frame.children_residents.data,delta);
				if(result == 0u){
					depth_dec += 1;
				}
			}
		}

		//! Update the stack base once all other counters have been updated.
		if(util::current_leader()){
			result = atomicSub(&(_dev_ctx.stack->depth_live),depth_dec);
			if(result == 0){
				set_flags(_grp_ctx,COMPLETION_FLAG);
			}
		}

	}


	//! Repetitively tries to merge the given queue of promises with the queue at the given index in the
	//! frame at the given level on the stack. This function currently aborts if an error flag is set
	//! or if too many merge failures occur, however, once its correctness is verified, this function
	//! will run forever until the merge is successful, as success is essentially guaranteed by
	//! the nature of the process.
	 __device__  void push_promises(unsigned int level, unsigned int index, QueueType queue, int promise_delta) {


		LinkAdrType tail = queue.get_tail();
		LinkAdrType head = queue.get_head();
		rc_printf("SM %d: push_promises(level:%d,index:%d,queue:(%d,%d),delta:%d)\n",threadIdx.x,level,index,tail.adr,head.adr,promise_delta);
		//! Do not bother pushing a null queue if there is no delta to report
		if( ( ! queue.is_null() ) || (promise_delta != 0) ){

			//! Change the resident counter of the destination frame by the number of promises
			//! that have been added to or removed from the given queue
			FrameType &dest = get_frame(level);
			unsigned int old_count;
			unsigned int new_count;
			if(promise_delta >= 0) {
				old_count = atomicAdd(&dest.children_residents.data,(PromiseCountPair) promise_delta);
				new_count = old_count + (unsigned int) promise_delta;
				if(old_count > new_count){
					rc_printf("\n\nOVERFLOW\n\n");
				}
			} else {
				PromiseCountPair neg_delta = -((PromiseCountPair) (-promise_delta));
				old_count = atomicAdd(&dest.children_residents.data,neg_delta);
				new_count = old_count - neg_delta;
				if(old_count < new_count){
					rc_printf("\n\nUNDERFLOW\n\n");
				}
			}


			_grp_ctx.SM_promise_delta += promise_delta;
			rc_printf("SM %d-%d: Old count: %d, New count: %d, Delta: %d\n",blockIdx.x,threadIdx.x,old_count,new_count,promise_delta);


			rc_printf("SM %d-%d: frame zero resident count is: %d\n",blockIdx.x,threadIdx.x,_dev_ctx.stack->frames[0].children_residents.data);
			//! If the addition caused a frame to change from empty to non-empty or vice-versa,
			//! make an appropriate incrementation or decrementation at the stack base.
			if( (old_count == 0) && (new_count != 0) ){
				atomicAdd(&(_dev_ctx.stack->depth_live),0x00010000u);
			} else if( (old_count != 0) && (new_count == 0) ){
				atomicSub(&(_dev_ctx.stack->depth_live),0x00010000u);
			} else {
				rc_printf("SM %d: No change!\n",threadIdx.x);
			}

			//! Finally, push the queues
			push_queue(dest.pool.queues[index],queue);
			rc_printf("SM %d: Pushed queue (%d,%d) to stack at index %d\n",threadIdx.x,queue.get_head().adr,queue.get_tail().adr,index);

			if( (_dev_ctx.stack->frames[0].children_residents.get_right()) == 0 ) {
				rc_printf("SM %d-%d: After queue pushed to stack, frame zero resident count is: %d\n",blockIdx.x,threadIdx.x,_dev_ctx.stack->frames[0].children_residents.data);
			}

		}
		rc_printf("(%d) the delta: %d\n",threadIdx.x,promise_delta);

	}



	//! Attempts to pull a queue of promises from the frame in the stack of the given level, starting the
	//! pull attempt at the given index in the frame. If no queue could be pulled after attempting a
	//! pull at each queue in the given frame, a QueueType::null value is returned.
	 __device__  QueueType pull_promises(unsigned int level, unsigned int& source_index) {


		rc_printf("SM %d: pull_promises(level:%d)\n",threadIdx.x,level);
		unsigned int src_idx = util::random_uint(_thd_ctx.rand_state) % FRAME_SIZE;

		__threadfence();
		FrameType &src = get_frame(level);

		QueueType queue = pull_queue(src.pool.queues,src_idx,FRAME_SIZE,source_index);

		if( ! queue.is_null() ){
			q_printf("SM %d: Pulled queue (%d,%d) from stack at index %d\n",blockIdx.x,queue.get_tail().adr,queue.get_head().adr, src_idx);
		} else {
			q_printf("SM %d: Failed to pull queue from stack starting at index %d\n",blockIdx.x, src_idx);
		}
		__threadfence();
		return queue;

	}


	//! Attempts to pull a queue from any frame in the stack, starting from the highest and working
	//! its way down. If no queue could be pulled, a QueueType::null value is returned.
	 __device__  QueueType pull_promises_any_level(unsigned int& level, unsigned int& source_index){


		QueueType result;
		result.data = QueueType::null;
		unsigned int start_level = highest_level();
		for(int level_itr = start_level; level_itr>=0; level_itr--){
			q_printf("Pulling promises at level %d for pull_promises_any_level\n",level_itr);
			result = pull_promises(level_itr,source_index);
			if( ! result.is_null() ){
				level = level_itr;
				return result;
			}
		}
		result.data = QueueType::null;
		return result;

	}





	//! Adds the contents of the stash slot at the given index to a link and returns the index of the
	//! link in the arena. This should only ever be called if there is both a link available to store
	//! the data and if the index is pointing at a non-empty slot. This also should only ever be
	//! called in a single-threaded context.
	 __device__  LinkAdrType produce_link(unsigned int slot_index ){


		__shared__ LinkAdrType result;

		unsigned int active = __activemask();


		//__syncwarp(active);

		//if(util::current_leader()){
			LinkAdrType link_index = claim_stash_link();
			q_printf("Claimed link %d from stash\n",link_index.adr);
			LinkType& the_link = _dev_ctx.arena[link_index];
			//_grp_ctx.SM_promise_delta += _grp_ctx.stash[slot_index].count;
			_grp_ctx.stash[slot_index].next = LinkAdrType::null;
			the_link = _grp_ctx.stash[slot_index];
			db_printf("Link has count %d and next %d in main memory",the_link.count, the_link.next);
			result = link_index;
			_grp_ctx.main_queue.count -= 1;
		//}


		//__syncwarp(active);
		return result;

	}








	//! Removes all promises in the stash that do not correspond to the given level, or to the levels
	//! immediately above or below (level+1) and (level-1).
	 __device__  void relevel_stash(unsigned int level){

		if( ! StackType::FLAT ){

			// TODO: Implement for non-flat stack

		}

	}





	//! Dumps all full links not corresponding to the current execution level. Furthermore, should the
	//! remaining links still put the stash over the given threshold occupancy, links will be further
	//! removed in the order: full links at the current level, partial links not at the current level,
	//! partial links at the current level.
	 __device__  void spill_stash(unsigned int threashold){

		unsigned int active =__activemask();
		__syncwarp(active);



	#if DEF_STACK_MODE == 0



		if(util::current_leader() && (_grp_ctx.main_queue.count > threashold)){


			unsigned int spill_count = _grp_ctx.main_queue.count - threashold;
			int delta = 0;
			fill_stash_links(spill_count);

			QueueType queue;
			queue.pair.data = QueueType::null;
			unsigned int partial_iter = 0;
			bool has_full_slots = true;
			q_printf("{Spilling to %d}",threashold);
			for(unsigned int i=0; i < spill_count; i++){
				unsigned int slot = STASH_SIZE;
				if(has_full_slots){
					slot = claim_full_slot();
					if(slot == STASH_SIZE){
						has_full_slots = false;
					}
				}
				if(! has_full_slots){
					for(;partial_iter < FN_ID_COUNT; partial_iter++){
						db_printf("%d",partial_iter);
						if(_grp_ctx.main_queue.partial_map[partial_iter] != STASH_SIZE){
							slot = _grp_ctx.main_queue.partial_map[partial_iter];
							q_printf("{Spilling partial}");
							partial_iter++;
							break;
						}
					}
				}
				if(slot == STASH_SIZE){
					break;
				}

				delta += _grp_ctx.stash[slot].count;
				q_printf("Slot for production (%d) has %d promises\n",slot,_grp_ctx.stash[slot].count);
				LinkAdrType link = produce_link(slot);
				push_back(queue,link);
				insert_empty_slot(slot);
				if(_grp_ctx.main_queue.count <= threashold){
					break;
				}
			}

			unsigned int push_index = util::random_uint(_thd_ctx.rand_state)%FRAME_SIZE;
			q_printf("Pushing promises in (%d,%d) for spilling\n",queue.get_head().adr,queue.get_tail().adr);
			push_promises(0,push_index,queue,delta);
			q_printf("Pushed queue (%d,%d) to stack\n",queue.get_head().adr,queue.get_tail().adr);


		}


	#else

		__shared__ unsigned int queues[3];
		__shared__ unsigned int counts[3];
		__shared__ unsigned int bucket[4];

		/*
		// Currently implemented in a single-threaded manner per work group to simplify the initial
		// correctness checking process. This can later be changed to take advantage of in-group
		// parallelism.
		*/
		if(util::current_leader()){

			/*
			// Zero the counters and null the queues
			*/
			for(unsigned int i=0; i < 3; i++){
				queues[i] = QueueType::null;
				counts[i] = 0;
			}

			for(unsigned int i=0; i < 4; i++){
				bucket[i] = 0;
			}


			/*
			// Count up each type of link
			*/
			for(unsigned int i=0; i < STASH_SIZE; i++){
				unsigned int depth = _grp_ctx.stash[i].depth;
				unsigned int size = _grp_ctx.stash[i].size;
				unsigned int idx = (depth != level) ? 0 : 1;
				idx += (size >= WARP_COUNT) ? 0 : 2;
				bucket[idx] += 1;
			}

			/*
			// Determine how much of which type of link needs to be dumped
			*/
			unsigned int dump_total = bucket[0];
			unsigned int dump_count = (_grp_ctx.main_queue.count > threshold) ? _grp_ctx.main_queue.count - threshold : 0;
			dump_count = (dump_count <= bucket[0]) : 0 ? dump_count - bucket[0];
			for(unsigned int i=1; i< 4; i++){
				unsigned int delta = (bucket[i] <= dump_count) ? bucket[i] : dump_count;
				dump_count -= delta;
				bucket[i] = delta;
				dump_total += delta;
			}

			/*
			// Dump the corresponding number of each type of link
			*/
			for(unsigned int i=0; i < _grp_ctx.main_queue.count; i++){
				unsigned int depth = _grp_ctx.stash[i].depth;
				unsigned int size  = _grp_ctx.stash[i].size;
				unsigned int bucket_idx = (depth != level) ? 0 : 1;
				bucket_idx += (size >= WARP_COUNT) ? 0 : 2;
				if(bucket[bucket_idx] == 0){
					continue;
				}
				LinkAdrType link = _grp_ctx.link_stash[_grp_ctx.link_stash_count];
				_grp_ctx.link_stash_count -= 1;

				copy_link(_dev_ctx.arena[link], _grp_ctx.stash[i]);

				unsigned int level_index = level+1-depth;
				counts[level_index] += size;
				push_back(queues[level_index],link);

				_grp_ctx.stash[i].size = 0;
			}
		}

	#endif

		__syncwarp(active);



	}


	//! Spills from stash in advance of inserting more work
	 __device__  void async_call_stash_dump(OpDisc func_id, int depth_delta, unsigned int delta){

		/*
		// Make room to queue incoming promises, if there isn't enough room already.
		*/
		#if 0
		if(_grp_ctx.link_stash_count < 2){
			fill_stash_links(2);
		}

		if(_grp_ctx.main_queue.count >= (STASH_SIZE-2)){
			spill_stash(STASH_SIZE-3);
		}
		#else
		/*
		unsigned int depth = (unsigned int) (_grp_ctx.level + depth_delta);
		unsigned int left_jump = partial_map_index(func_id,depth,_grp_ctx.level);
		unsigned int space = 0;
		if( left_jump != PART_ENTRY_COUNT ){
			unsigned int left_idx = _grp_ctx.main_queue.partial_map[left_jump];
			if( left_idx != STASH_SIZE ){
				space = WORK_GROUP_SIZE - _grp_ctx.stash[left_idx].count;
			}
		}
		*/
		if( stash_overfilled() ) { //&& (space < delta) ){
			if(_grp_ctx.link_stash_count < STASH_MARGIN){
				fill_stash_links(STASH_MARGIN);
			}
			//printf("{Spilling for call.}");
			spill_stash(STASH_HIGH_WATER-1);
		}
		#endif

	}


	//! Prepares the stash for the insertion of work
	 __device__  void async_call_stash_prep(RemapQueue& dst, OpDisc func_id, int depth_delta, unsigned int delta,
		unsigned int &left, unsigned int &left_start, unsigned int &right
	){

		/*
		// Locate the destination links in the stash that the promises will be written to. For now,
		// like many other parts of the code, this will be single-threaded within the work group
		// to make validation easier but will be optimized for group-level parallelism later.
		*/
		if( util::current_leader() ){

			db_printf("{Queueing %d promises of type %d}",delta,func_id);
			/*
			// Null out the right index. This index should not be used unless the number of
			// promises queued spills over beyond the first link being written to (the left one)
			*/
			right = STASH_SIZE;

			/*
			// Find the index of the partial link in the stash corresponding to the id and
			// depth of the calls being queued (if it exists).
			*/
			unsigned int depth = (unsigned int) (_grp_ctx.level + depth_delta);
			unsigned int left_jump = partial_map_index(func_id,depth,_grp_ctx.level);

			/*
			// If there is a partially filled link to be filled, assign that to the left index
			*/
			if(left_jump != PART_ENTRY_COUNT){
				db_printf("(%d:A)",blockIdx.x);
				left = dst.partial_map[left_jump];
			}

			unsigned int left_count;
			if(left == STASH_SIZE){
				left = claim_empty_slot();
				dst.count += 1;
				db_printf("(%d:B+1->%d)",blockIdx.x,dst.count);
				db_printf("Updated stash count: %d\n",dst.count);
				_grp_ctx.stash[left].id    = func_id;
				dst.partial_map[left_jump] = left;
				left_count = 0;
			} else {
				left_count = _grp_ctx.stash[left].count;
			}

			if ( (left_count + delta) > WORK_GROUP_SIZE ){
				right = claim_empty_slot();
				dst.count += 1;
				db_printf("(%d:C+1->%d)",blockIdx.x,dst.count);
				_grp_ctx.stash[right].count = left_count+delta - WORK_GROUP_SIZE;
				_grp_ctx.stash[right].id    = func_id;
				insert_full_slot(dst,left);
				dst.partial_map[left_jump] = right;
				_grp_ctx.stash[left].count = WORK_GROUP_SIZE;
			} else if ( (left_count + delta) == WORK_GROUP_SIZE ){
				db_printf("(%d:D)",blockIdx.x);
				dst.partial_map[left_jump] = STASH_SIZE;
				insert_full_slot(dst,left);
				_grp_ctx.stash[left].count = WORK_GROUP_SIZE;
			} else {
				db_printf("(%d:E)",blockIdx.x);
				_grp_ctx.stash[left].count = left_count + delta;
			}

			left_start = left_count;


		}

	}


	/*
	// Queues the input promise into a corresponding local link according to the given function id and
	// at a level corresponding to the current level of he promises being evaluated plus the value of
	// depth_delta. This scheme ensures that the function being called and the depth of the promises
	// being created for those calls are consistent across the warp.
	*/
	 __device__  void async_call(OpDisc func_id, int depth_delta, PromiseUnionType& promise){

		unsigned int active = __activemask();

		/*
		// Calculate how many promises are being queued as well as the assigned index of the
		// current thread's promise in the write to the stash.
		*/
		unsigned int index = util::warp_inc_scan();
		unsigned int delta = util::active_count();


		#ifdef BARRIER_SPILL
		if( stash_overfilled() ) { //&& (space < delta) ){
			_grp_ctx.spill_barrier.union_await<false>(*this,func_id,promise);
			return;
		}
		#else
		async_call_stash_dump(func_id, depth_delta, delta);
		#endif


		__shared__ unsigned int left, left_start, right;


		async_call_stash_prep(_grp_ctx.main_queue,func_id,depth_delta,delta,left,left_start,right);


		/*
		// Write the promise into the appropriate part of the stash, writing into the left link
		// when possible and spilling over into the right link when necessary.
		*/
		__syncwarp(active);
		if( (left_start + index) >= WORK_GROUP_SIZE ){
			//db_printf("Overflow: id: %d, left: %d, left_start: %d, index: %d\n",threadIdx.x,left,left_start,index);
			_grp_ctx.stash[right].promises[left_start+index-WORK_GROUP_SIZE].dyn_copy_as(func_id,promise);
			//_grp_ctx.stash[right].promises[left_start+index-WORK_GROUP_SIZE] = promise;
		} else {
			//db_printf("Non-overflow: id: %d, left: %d, left_start: %d, index: %d\n",threadIdx.x,left,left_start,index);
			_grp_ctx.stash[left].promises[left_start+index].dyn_copy_as(func_id,promise);
			//_grp_ctx.stash[left].promises[left_start+index] = promise;
		}
		__syncwarp(active);

	}

	/*
	// Like async_call, but allows for one to hand in the underlying type corresponding to a function id directly
	*/
	template<typename TYPE, typename... ARGS>
	 __device__  void async_call_cast(int depth_delta, ARGS... args){

		beg_time(7);
		unsigned int active = __activemask();

		/*
		// Calculate how many promises are being queued as well as the assigned index of the
		// current thread's promise in the write to the stash.
		*/
		unsigned int index = util::warp_inc_scan();
		unsigned int delta = util::active_count();

		#ifdef BARRIER_SPILL
		if( stash_overfilled() ) { //&& (space < delta) ){
			beg_time(8);
			_grp_ctx.spill_barrier.await<false>(*this,Promise<TYPE>(args...));
			end_time(8);
			return;
		}
		#else
		beg_time(8);
		async_call_stash_dump(Lookup<TYPE>::type::DISC, depth_delta, delta);
		end_time(8);
		#endif

		__shared__ unsigned int left, left_start, right;


		#ifdef ASYNC_LOADS
		RemapQueue& dst_queue = has_load(args...) ? _grp_ctx.load_queue : _grp_ctx.main_queue;
		#else
		RemapQueue& dst_queue = _grp_ctx.main_queue;
		#endif

		beg_time(9);
		async_call_stash_prep(dst_queue,Lookup<TYPE>::type::DISC,depth_delta,delta,left,left_start,right);
		end_time(9);

		/*
		// Write the promise into the appropriate part of the stash, writing into the left link
		// when possible and spilling over into the right link when necessary.
		*/
		__syncwarp(active);
		if( (left_start + index) >= WORK_GROUP_SIZE ){
			//db_printf("Overflow: id: %d, left: %d, left_start: %d, index: %d\n",threadIdx.x,left,left_start,index);
			#ifdef ASYNC_LOADS
			_grp_ctx.stash[right].promises[left_start+index-WORK_GROUP_SIZE].template cast<TYPE>().async_load(*this,args...);
			#else
			_grp_ctx.stash[right].promises[left_start+index-WORK_GROUP_SIZE].template cast<TYPE>() = Promise<TYPE>(args...);
			#endif
		} else {
			//db_printf("Non-overflow: id: %d, left: %d, left_start: %d, index: %d\n",threadIdx.x,left,left_start,index);
			#ifdef ASYNC_LOADS
			_grp_ctx.stash[left].promises[left_start+index].template cast<TYPE>().async_load(*this,args...);
			#else
			_grp_ctx.stash[left].promises[left_start+index].template cast<TYPE>() = Promise<TYPE>(args...);
			#endif
		}
		__syncwarp(active);
		end_time(7);

	}


	template<typename TYPE>
	 __device__  void async_call_cast(int depth_delta, Promise<TYPE> promise){
		beg_time(7);
		unsigned int active = __activemask();

		/*
		// Calculate how many promises are being queued as well as the assigned index of the
		// current thread's promise in the write to the stash.
		*/
		unsigned int index = util::warp_inc_scan();
		unsigned int delta = util::active_count();




		#ifdef BARRIER_SPILL
		if( stash_overfilled() ) { //&& (space < delta) ){
			beg_time(8);
			_grp_ctx.spill_barrier.await<false>(*this,promise);
			end_time(8);
			return;
		}
		#else
		beg_time(8);
		async_call_stash_dump(Lookup<TYPE>::type::DISC, depth_delta, delta);
		end_time(8);
		#endif

		__shared__ unsigned int left, left_start, right;


		beg_time(9);
		async_call_stash_prep(_grp_ctx.main_queue,Lookup<TYPE>::type::DISC,depth_delta,delta,left,left_start,right);
		end_time(9);

		/*
		// Write the promise into the appropriate part of the stash, writing into the left link
		// when possible and spilling over into the right link when necessary.
		*/
		__syncwarp(active);
		if( (left_start + index) >= WORK_GROUP_SIZE ){
			//db_printf("Overflow: id: %d, left: %d, left_start: %d, index: %d\n",threadIdx.x,left,left_start,index);
			_grp_ctx.stash[right].promises[left_start+index-WORK_GROUP_SIZE].template cast<TYPE>() = promise;
		} else {
			//db_printf("Non-overflow: id: %d, left: %d, left_start: %d, index: %d\n",threadIdx.x,left,left_start,index);
			_grp_ctx.stash[left].promises[left_start+index].template cast<TYPE>() = promise;
		}
		__syncwarp(active);
		end_time(7);

	}


	template<typename TYPE>
	__device__  void immediate_call_cast(Promise<TYPE> param_value){
		PromiseUnionType promise;
		promise.template cast<TYPE>() = param_value;
		promise.template rigid_eval<ProgramType,TYPE>(*this);
		//promise_eval<ProgramType,FUNC_ID>(param_value);

	}



	#define PARACON

	/*
	// Adds the contents of the link at the given index to the stash and adds the given link to link
	// stash. Once complete, it returns the number of promises added to the stash by the operation.
	// This should only ever be called if there is enough space to store the extra work and link.
	*/
	 __device__  unsigned int consume_link(LinkAdrType link_index ){


		#if 0 //def PARACON
		__shared__ LinkAdrType the_index;
		__shared__ unsigned int add_count;
		__shared__ OpDisc func_id;

		unsigned int active = __activemask();

		__syncwarp(active);

		if(util::current_leader()){

			q_printf("Consuming link %d\n",link_index.adr);

			the_index = link_index;
			add_count = _dev_ctx.arena[link_index].count;
			func_id   = _dev_ctx.arena[link_index].id;

		}


		__syncwarp(active);

		#if 0
		if(threadIdx.x < add_count){
			async_call(func_id,0,_dev_ctx.arena[the_index].promises[threadIdx.x]);
		}
		#else
		unsigned int idx = util::warp_inc_scan();
		unsigned int tot = util::active_count();
		for(unsigned int i=idx; i<add_count; i+=tot){
			async_call(func_id,0,_dev_ctx.arena[the_index].promises[i]);
		}
		#endif


		__syncwarp(active);


		if(util::current_leader()){
			insert_stash_link(link_index);
		}

		return add_count;


		#else

		LinkAdrType the_index;
		unsigned int add_count;
		OpDisc       func_id;

		unsigned int active = __activemask();
		unsigned int acount = util::active_count();



		the_index = link_index;
		add_count = _dev_ctx.arena[link_index].count;
		func_id   = _dev_ctx.arena[link_index].id;

		//_grp_ctx.SM_promise_delta -= add_count;

		db_printf("active count: %d, add count: %d\n",acount,add_count);


		db_printf("\n\nprior stash count: %d\n\n\n",_grp_ctx.main_queue.count);
		//*
		for(unsigned int i=0; i< add_count; i++){
			//PromiseUnionType promise = _dev_ctx.arena[the_index].promises[i];
			//async_call(func_id,0,promise);
			async_call(func_id,0, _dev_ctx.arena[the_index].promises[i] );
		}
		// */
		//PromiseType promise = _dev_ctx.arena[the_index].data.data[0];
		//async_call(func_id,0,promise);

		db_printf("\n\nafter stash count: %d\n\n\n",_grp_ctx.main_queue.count);


		insert_stash_link(link_index);

		return add_count;



		#endif


	}






	/*
	// Tries to transfer links from the stack into the stash of the work group until the stash
	// is filled to the given threashold. If a halting condition is reached, this function will set
	// the keep_running value in the shared context to false.
	*/
	 __device__  void fill_stash(unsigned int threashold, bool halt_on_fail){

		unsigned int active =__activemask();
		__syncwarp(active);


		#ifdef PARACON
		__shared__ unsigned int link_count;
		__shared__ LinkAdrType links[STASH_SIZE];
		#endif

		/*
		// Currently implemented in a single-threaded manner per work group to simplify the initial
		// correctness checking process. This can later be changed to take advantage of in-group
		// parallelism.
		*/
		if(util::current_leader()){

			//db_printf("Filling stash...\n");

			unsigned int taken = 0;

			beg_time(12);
			threashold = (threashold > STASH_SIZE) ? STASH_SIZE : threashold;
			//printf("{Filling to %d @ %d}",threashold,blockIdx.x);

			unsigned int gather_count = (threashold < _grp_ctx.main_queue.count) ? 0  : threashold - _grp_ctx.main_queue.count;
			if( (STASH_SIZE - _grp_ctx.link_stash_count) < gather_count){
				unsigned int spill_thresh = STASH_SIZE - gather_count;
				spill_stash_links(spill_thresh);
			}
			end_time(12);


			#ifdef PARACON
			unsigned int _thd_ctx_link_count = 0;
			#endif

			#ifdef RACE_COND_PRINT
			unsigned int p_depth_live = _dev_ctx.stack->depth_live;
			rc_printf("SM %d: depth_live is (%d,%d)\n",threadIdx.x,(p_depth_live&0xFFFF0000)>>16,p_depth_live&0xFFFF);
			#endif

			for(unsigned int i = 0; i < FILL_STASH_RETRY_LIMIT; i++){

				/* If the stack is empty or a flag is set, return false */
				unsigned int depth_live = _dev_ctx.stack->depth_live;
				if( (depth_live == 0u) || ( _dev_ctx.stack->status_flags != 0u) ){
					if( halt_on_fail || ( _dev_ctx.stack->status_flags != 0u) ) {
						_grp_ctx.keep_running = false;
					}
					break;
				}


				unsigned int src_index;
				QueueType queue;

				beg_time(3);
				#if DEF_STACK_MODE == 0

				db_printf("STACK MODE ZERO\n");
				q_printf("%dth try pulling promises for fill\n",i+1);
				if( get_frame(_grp_ctx.level).children_residents.data != 0 ){
					queue = pull_promises(_grp_ctx.level,src_index);
				} else {
					queue.pair.data = QueueType::null;
				}

				#else
				/*
				// Determine whether or not to pull from the current level in the stack
				*/
				unsigned int depth = left_half(depth_live);
				bool pull_any = (depth < _grp_ctx.level);
				FrameType &current_frame = get_frame(depth);
				if(!pull_any){
					pull_any = ((current_frame.children_residents.get_right()) == 0);
				}


				/*
				// Retrieve a queue from the stack.
				*/

				if(pull_any){
					unsigned int new_level;
					queue = pull_promises_any_level(new_level,src_index);
					relevel_stash(new_level);
				} else {
					queue = pull_promises(_grp_ctx.level,src_index);
				}
				#endif
				end_time(3);


				beg_time(11);
				#ifdef PARACON
				db_printf("About to pop promises\n");
				while(	( ! queue.is_null() )
				     && (_thd_ctx_link_count < gather_count)
				     && (_grp_ctx.link_stash_count < STASH_SIZE)
				){
					beg_time(13);
					LinkAdrType link = pop_front(queue);
					end_time(13);
					if( ! link.is_null() ){
						beg_time(14);
						db_printf("Popping front %d\n",link);
						links[_thd_ctx_link_count] = link;
						taken += _dev_ctx.arena[link].count;
						_thd_ctx_link_count++;
						end_time(14);
					} else {
						break;
					}
				}
				#else
				db_printf("About to pop promises\n");
				while(	( ! queue.is_null() )
				     && (_grp_ctx.main_queue.count < threashold)
				     && (_grp_ctx.link_stash_count < STASH_SIZE)
				){
					beg_time(13);
					LinkAdrType link = pop_front(queue);
					end_time(13);

					q_printf("Popping front %d. Q is now (%d,%d)\n",link.adr,queue.get_head().adr,queue.get_tail().adr);

					if( ! link.is_null() ){
						beg_time(14);
						taken += consume_link(link);
						end_time(14);
					} else {
						break;
					}
				}
				#endif
				end_time(11);

				db_printf("Popped promises\n");
				if(taken != 0){
					if(!_grp_ctx.busy){
						atomicAdd(&(_dev_ctx.stack->depth_live),1);
						_grp_ctx.busy = true;
						//printf("{got busy %d depth_live=(%d,%d)}",blockIdx.x,(depth_live & 0xFFFF0000)>>16u, depth_live & 0xFFFF);
						rc_printf("SM %d: Incremented depth value\n",threadIdx.x);
					}
					rc_printf("Pushing promises for filling\n");
					push_promises(_grp_ctx.level,src_index,queue,-taken);
					break;
				}

				#ifdef PARACON
				if( _thd_ctx_link_count >= gather_count ){
					break;
				}
				#else
				if( _grp_ctx.main_queue.count >= threashold ){
					break;
				}
				#endif

			}





			#ifdef PARACON
			if(_grp_ctx.busy && (_grp_ctx.main_queue.count == 0) && (taken == 0) ){
			#else
			if(_grp_ctx.busy && (_grp_ctx.main_queue.count == 0)){
			#endif
				unsigned int depth_live = atomicSub(&(_dev_ctx.stack->depth_live),1);
				//printf("{unbusy %d depth_live=(%d,%d)}",blockIdx.x,(depth_live & 0xFFFF0000)>>16u, depth_live & 0xFFFF);
				rc_printf("SM %d: Decremented depth value\n",threadIdx.x);
				_grp_ctx.busy = false;
			}


			#ifdef PARACON
			link_count = _thd_ctx_link_count;
			#endif


		}

		__syncwarp(active);



		#ifdef PARACON

		__threadfence();
		beg_time(15);
		if(util::current_leader()){
			for(int i=0; i<link_count;i++){
				consume_link(links[i]);
			}
		}
		end_time(15);


		__syncwarp(active);
		__threadfence();
		#endif



	}




	 __device__  void clear_exec_head(){


		if( util::current_leader() && (_grp_ctx.exec_head != STASH_SIZE) ){
			insert_empty_slot(_grp_ctx.exec_head);
			_grp_ctx.exec_head = STASH_SIZE;
		}
		__syncwarp();

	}




	/*
	// Selects the next link in the stash. This selection process could become more sophisticated
	// in later version to account for the average branching factor of each async function. For now,
	// it selects the fullest slot of the current level if it can. If no slots with promises for the
	// current level exist in the stash, the function returns false.
	*/
	 __device__  bool advance_stash_iter(){

		__shared__ bool result;
		unsigned int active =__activemask();
		__syncwarp(active);


		if(util::current_leader()){

			if(_grp_ctx.main_queue.full_head != STASH_SIZE){
				_grp_ctx.exec_head = claim_full_slot();
				_grp_ctx.main_queue.count -= 1;
				result = true;
				//db_printf("Found full slot.\n");
			} else {
				//db_printf("Looking for partial slot...\n");
				unsigned int best_id   = PART_ENTRY_COUNT;
				unsigned int best_slot = STASH_SIZE;
				unsigned int best_count = 0;
				for(int i=0; i < FN_ID_COUNT; i++){
					unsigned int slot = _grp_ctx.main_queue.partial_map[i];

					if( (slot != STASH_SIZE) && (_grp_ctx.stash[slot].count > best_count)){
						best_id = i;
						best_slot = slot;
						best_count = _grp_ctx.stash[slot].count;
					}

				}

				result = (best_slot != STASH_SIZE);
				if(result){
					//db_printf("Found partial slot.\n");
					_grp_ctx.exec_head = best_slot;
					_grp_ctx.main_queue.partial_map[best_id] = STASH_SIZE;
					_grp_ctx.main_queue.count -=1;
				}
			}

		}

		__syncwarp(active);
		return result;

	}


	#ifdef ASYNC_LOADS
	 __device__ void force_full_async_loads() {

		_grp_ctx.load_barrier.arrive_and_wait();
		__syncwarp();
		#if 1
		if (util::current_leader())  {
			unsigned char iter = _grp_ctx.load_queue.full_head;
			if( iter != STASH_SIZE ){
				unsigned int main_delta = 1;
				unsigned char next = _grp_ctx.stash[iter].next.adr;
				while( next != STASH_SIZE ){
					iter = next;
					next = _grp_ctx.stash[iter].next.adr;
					main_delta += 1;
				}
				_grp_ctx.stash[iter].next = _grp_ctx.main_queue.full_head;
				_grp_ctx.main_queue.full_head = _grp_ctx.load_queue.full_head;
				_grp_ctx.load_queue.full_head = STASH_SIZE;
				_grp_ctx.main_queue.count += main_delta;
				_grp_ctx.load_queue.count -= main_delta;
			}
			//printf("(%d:f-%d)",blockIdx.x,_grp_ctx.main_queue.count);
		}
		#else
		if( util::current_leader() ){
			_grp_ctx.main_queue.full_head = _grp_ctx.load_queue.full_head;
			_grp_ctx.load_queue.full_head = STASH_SIZE;
			_grp_ctx.main_queue.count += _grp_ctx.load_queue.count;
			_grp_ctx.load_queue.count = 0;
			printf("(%d:f-%d)",blockIdx.x,_grp_ctx.main_queue.count);
		}
		#endif
		__syncwarp();

	 }
	 __device__ void force_async_loads() {

		force_full_async_loads();

		for(int i=0; i < FN_ID_COUNT; i++){
			unsigned char idx = _grp_ctx.load_queue.partial_map[i];
			if( idx != STASH_SIZE){
				LinkType& link = _grp_ctx.stash[idx];
				if( threadIdx.x < link.count ){
					async_call(0,link.id,link.promises[threadIdx.x]);
				}
				__syncwarp();
				if( util::current_leader() ){
					insert_empty_slot(idx);
					_grp_ctx.load_queue.partial_map[i] = STASH_SIZE;
					//printf("(%d:%d)\n",blockIdx.x,idx);
				}
			}
		}
		__syncwarp();
		if(util::current_leader()){
			init(&_grp_ctx.load_barrier,WORK_GROUP_SIZE);
			_grp_ctx.load_queue.count = 0;
		}

	 }
	#endif



	/*
	// Tries to perform up to one work group worth of work by selecting a link from shared memory (or,
	// if necessary, fetching a link from main memory), and running the function on the data within
	// the link, as directed by the function id the link is labeled with. This function returns false
	// if a halting condition has been reached (either due to lack of work or an event) and true
	// otherwise.
	*/
	 __device__  void exec_cycle(){



		clear_exec_head();

		/*
		// Advance the stash iterator to the next chunk of work that needs to be done.
		*/
		//*


		if( stash_overfilled() ) {
			//printf("{Stash overfilled}");
			#ifdef BARRIER_SPILL
			if( util::current_leader() ){
				_grp_ctx.spill_barrier.release_full(*this);
			}
			#endif

			if(_grp_ctx.link_stash_count < STASH_MARGIN){
				fill_stash_links(STASH_MARGIN);
			}
			//printf("{Spilling for call.}");
			spill_stash(STASH_HIGH_WATER-1);
		}


		beg_time(1);

		#ifdef ASYNC_LOADS
		if( (_grp_ctx.main_queue.full_head == STASH_SIZE) && (_grp_ctx.load_queue.full_head != STASH_SIZE) ){
			force_full_async_loads();
		}
		#endif

		const PromiseCount GLOBAL_WORK_THRESHOLD = STASH_SIZE/2;
		//if ( ( ((_dev_ctx.stack->frames[0].children_residents) & 0xFFFF ) > (gridDim.x*blockIdx.x*2) ) && (_grp_ctx.main_queue.full_head == STASH_SIZE) ) {
		if ( ( ((_dev_ctx.stack->frames[0].children_residents.get_right()) ) > GLOBAL_WORK_THRESHOLD ) && (_grp_ctx.main_queue.full_head == STASH_SIZE) ) {
			fill_stash(STASH_HIGH_WATER,false);
		}

		end_time(1);

		beg_time(5);
		while ( _grp_ctx.can_make_work && (_grp_ctx.main_queue.full_head == STASH_SIZE) ) {
			_grp_ctx.can_make_work = __any_sync(0xFFFFFFFF,PROGRAM_SPEC::make_work(*this));
			if( util::current_leader() && (! _grp_ctx.busy ) && ( _grp_ctx.main_queue.count != 0 ) ){
				unsigned int depth_live = atomicAdd(&(_dev_ctx.stack->depth_live),1);
				_grp_ctx.busy = true;
				//printf("{made self busy %d depth_live=(%d,%d)}",blockIdx.x,(depth_live & 0xFFFF0000)>>16u, depth_live & 0xFFFF);
			}
			__syncwarp();
		}
		end_time(5);

		#ifdef ASYNC_LOADS
		if( (_grp_ctx.main_queue.full_head == STASH_SIZE) && (_grp_ctx.load_queue.count > 0) ){
			force_async_loads();
		}
		#endif

		#if 1

		/*
		if ( _grp_ctx.main_queue.full_head == STASH_SIZE ) {
			fill_stash(STASH_SIZE-2);
		}
		*/
		#else
		if(_grp_ctx.main_queue.full_head == STASH_SIZE){
			if( !_grp_ctx.scarce_work ){
				fill_stash(STASH_SIZE-2);
				if( util::current_leader() && (_grp_ctx.main_queue.full_head == STASH_SIZE) ){
					_grp_ctx.scarce_work = true;
				}
			}
		} else {
			if( util::current_leader() ){
				_grp_ctx.scarce_work = false;
			}
		}
		#endif
		// */

		#if 0
		if( util::current_leader() ){
			PairPack<PromiseCount> cr = _dev_ctx.stack->frames[0].children_residents;
			printf("{group %d children_residents=(%d,%d), can_make_work=%d}\n",blockIdx.x,cr.get_left(),cr.get_right(),_grp_ctx.can_make_work);
		}
		#endif
		beg_time(2);
		if( !advance_stash_iter() ){
			/*
			// No more work exists in the stash, so try to fetch it from the stack.
			*/
			beg_time(10);
			fill_stash(STASH_HIGH_WATER,true);
			end_time(10);

			if( _grp_ctx.keep_running && !advance_stash_iter() ){
				/*
				// REALLY BAD: The fill_stash function was successful, however
				// the stash still has no work to perform. In this situation,
				// we set an error flag and halt.
				*/
				/*
				if(util::current_leader()){
					db_printf("\nBad stuff afoot!\n\n");
				}
				set_flags(_grp_ctx,STASH_FAIL_FLAG);
				_grp_ctx.keep_running = false;
				*/
			}
		}
		end_time(2);


		unsigned int active = __activemask();
		__syncwarp(active);


		beg_time(4);
		if( _grp_ctx.exec_head != STASH_SIZE ){
			/*
			// Find which function the current link corresponds to.
			*/
			OpDisc func_id     = _grp_ctx.stash[_grp_ctx.exec_head].id;
			unsigned int promise_count = _grp_ctx.stash[_grp_ctx.exec_head].count;

			/*
			// Only execute if there is a promise in the current link corresponding to the thread that
			// is being executed.
			*/
			if(util::current_leader()){
				db_printf("Executing slot %d, which is %d promises of type %d\n",_grp_ctx.exec_head,promise_count,func_id);
			}
			if( threadIdx.x < promise_count ){
				//db_printf("Executing...\n");
				PromiseUnionType& promise = _grp_ctx.stash[_grp_ctx.exec_head].promises[threadIdx.x];
				//do_async(func_id,promise);
				promise.template loose_eval(*this,func_id);
			}
		}

		__syncwarp(active);
		end_time(4);


	}



	 __device__  void cleanup_runtime(){


		//unsigned int active = __activemask();
		//__syncwarp(active);
		__syncwarp();


		if(_thd_ctx.spare_index > 0){
			dealloc_links(_thd_ctx.spare_links,_thd_ctx.spare_index);
		}

		if(threadIdx.x == 0){

			#ifdef BARRIER_SPILL
			_grp_ctx.spill_barrier.release(*this);
			#endif

			q_printf("CLEANING UP\n");
			clear_exec_head();

			spill_stash(0);
			spill_stash_links(0);

			if(_grp_ctx.can_make_work){
				//printf("{Setting early halt flag.}");
				set_flags(EARLY_HALT_FLAG);
			}

			if(_grp_ctx.busy){
				unsigned int depth_live = atomicSub(&(_dev_ctx.stack->depth_live),1);
				//printf("{wrap busy %d depth_live=(%d,%d)}",blockIdx.x,(depth_live & 0xFFFF0000)>>16u, depth_live & 0xFFFF);
			}
		}

		//__syncwarp(active);
		__syncwarp();
		__threadfence();
		//__syncwarp(active);
		__syncwarp();

		if(threadIdx.x == 0){
			unsigned int checkout_index = atomicAdd(&(_dev_ctx.stack->checkout),1);
			__threadfence();
			//printf("{%d}",checkout_index);
			if( checkout_index == (gridDim.x-1) ){
				//printf("{Final}\n");
				atomicExch(&(_dev_ctx.stack->checkout),0);
				unsigned int old_flags = atomicAnd(&(_dev_ctx.stack->status_flags),~EARLY_HALT_FLAG);
				unsigned int depth_live = atomicAdd(&(_dev_ctx.stack->depth_live),0);
				bool halted_early       = ( old_flags && EARLY_HALT_FLAG );
				bool work_left          = ( (depth_live & 0xFFFF0000) != 0 );

				if( (!halted_early) && (!work_left) ){
					set_flags(COMPLETION_FLAG);
				}

				//printf("{depth_live is (%d,%d)}",(depth_live&0xFFFF0000)>>16,depth_live&0xFFFF );
				//unsigned int cr = atomicAdd(&(_dev_ctx.stack->frames[0].children_residents),0);
				//printf("{Level 0 CR is (%d,%d)}",(cr&0xFFFF0000)>>16,cr&0xFFFF );
			}

			#ifdef HRM_TIME
			end_time(0);
			for(int i=0; i<HRM_TIME; i++){
				atomicAdd(&_dev_ctx.time_totals[i],_grp_ctx.time_totals[i]);
			}
			#endif

		}

	}


	public:
	/*
	//
	// This must be run once on the resources used for execution, prior to execution. Given that this
	// essentially wipes all data from these resources and zeros all values, it is not advised that
	// this function be used at any other time, except to setup for a re-start or to clear out after
	// calling the pull_runtime to prevent promise duplication.
	//
	*/
	__device__ void init_program(){

		/* Initialize per-thread resources */
		init_thread();

		const unsigned int threads_per_frame = FRAME_SIZE + 1;
		const unsigned int total_stack_work = StackType::NULL_LEVEL * threads_per_frame;

		unsigned int worker_count = gridDim.x*blockDim.x;

		/*
		// If the currently executing thread has device thread index 0, wipe the data in the base
		// of the stack.
		*/
		if(_thd_ctx.thread_id == 0){
			_dev_ctx.stack->status_flags = 0;
			_dev_ctx.stack->depth_live   = 0;
			_dev_ctx.stack->checkout     = 0;
		}


		/*
		if( _thd_ctx.thread_id == 0 ){
			printf(	"Initializing the stack with\n"
				"\t- total_stack_work=%d\n"
				"\t- threads_per_frame=%d\n"
				"\t- worker_count=%d\n"
				"\t- stack->frames[0].children_residents=%d\n",
				total_stack_work,
				threads_per_frame,
				worker_count,
				_dev_ctx.stack->frames[0].children_residents
			);
		}
		*/



		/*
		// Blank out the frames in the stack. Setting queues to NULL_QUEUE, and zeroing the counts
		// for resident promises and child promises of each frame.
		*/
		for(unsigned int index = _thd_ctx.thread_id; index < total_stack_work; index+=worker_count ){

			unsigned int target_level = index / threads_per_frame;
			unsigned int frame_index  = index % threads_per_frame;
			if( frame_index == FRAME_SIZE ){
				_dev_ctx.stack->frames[target_level].children_residents.data = 0u;
				if( _dev_ctx.stack->frames[target_level].children_residents.data != 0u ){
					printf("BAD A!\n");
				}
			} else {
				_dev_ctx.stack->frames[target_level].pool.queues[frame_index].pair.data = QueueType::null;
				if( _dev_ctx.stack->frames[target_level].pool.queues[frame_index].pair.data != QueueType::null ) {
					printf("BAD B!\n");
				}
			}

		}


		#ifdef LAZY_LINK

		if(_thd_ctx.thread_id == 0){
			(*_dev_ctx.claim_count) = 0;
		}

		/*
		// Initialize the pool, assigning empty queues to each queue slot.
		*/
		for(unsigned int index = _thd_ctx.thread_id; index < POOL_SIZE; index+=worker_count ){

			_dev_ctx.pool->queues[index].pair.data = QueueType::null;
			if( _dev_ctx.pool->queues[index].pair.data != QueueType::null ){
				printf("BAD C!\n");
			}

		}

		#else
		/*
		// Initialize the arena, connecting the contained links into roughly equally sized lists,
		// zeroing the promise counter in the links and marking the function ID with an invalid
		// value to make use-before-initialization more obvious during system validation.
		*/
		unsigned int bump = ((arena_size%POOL_SIZE) != 0) ? 1 : 0;
		unsigned int arena_init_stride = arena_size/POOL_SIZE + bump;
		for(unsigned int index = _thd_ctx.thread_id; index < arena_size; index+=worker_count ){

			unsigned int next = index + 1;
			if( ( (next % arena_init_stride) == 0 ) || (next >= arena_size) ){
				next = LinkAdrType::null;
			}
			_dev_ctx.arena[index].empty(LinkAdrType(next));
		}


		/*
		// Initialize the pool, giving each queue slot one of the previously created linked lists.
		*/
		for(unsigned int index = _thd_ctx.thread_id; index < POOL_SIZE; index+=worker_count ){

			unsigned int head = arena_init_stride * index;
			unsigned int tail = arena_init_stride * (index + 1) - 1;
			tail = (tail >= arena_size) ? arena_size - 1 : tail;
			_dev_ctx.pool->queues[index] = QueueType(LinkAdrType(head),LinkAdrType(tail));

		}
		#endif


	}


	/*
	// Unpacks all promise data from the call buffer into the stack of the given context. This
	// could be useful for backing up program states for debugging or to re-start processing from
	// a previous state.
	*/
	 __device__  void push_calls(DeviceContext _dev_ctx, LinkType* call_buffer, size_t link_count){

		/* Initialize per-warp resources */
		__shared__ GroupContext _grp_ctx;
		init_group(_dev_ctx,_grp_ctx);

		/* Initialize per-thread resources */
		ThreadContext _thd_ctx;
		init_local(_thd_ctx);


		for(int link_index=blockIdx.x; link_index < link_count; link_index+= gridDim.x){
			LinkType& the_link = call_buffer[link_index];
			unsigned int count   = the_link.count;
			unsigned int func_id = the_link.id;
			if(threadIdx.x < count){
				db_printf("\nasync_call(id:%d,depth: 0)\n\n",func_id);
				async_call(func_id,0,the_link.data.data[threadIdx.x]);
			}

		}

		cleanup_runtime();

	}




	 static void check_error(){

		cudaError_t status = cudaGetLastError();

		if(status != cudaSuccess){
			const char* err_str = cudaGetErrorString(status);
			printf("ERROR: \"%s\"\n",err_str);
		}

	}





	/*
	// Places a single function call into the runtime.
	*/
	 static void remote_call(Instance &instance, unsigned char func_id, PromiseUnionType promise){

		LinkType* call_buffer;
		cudaMalloc( (void**) &call_buffer, sizeof(LinkType) );

		LinkType host_link;
		host_link.count		= 1;
		host_link.id    	= func_id;
		host_link.next    	= LinkType::null;
		host_link.depth    	= 0;
		host_link.meta_data.data= 0;
		host_link.data.data[0]	= promise;

		cudaMemcpy(call_buffer,&host_link,sizeof(LinkType),cudaMemcpyHostToDevice);


		push_runtime<<<1,WORK_GROUP_SIZE>>>(instance.to_context(),call_buffer,1);

		check_error();

		cudaFree(call_buffer);

	}





	/*
	// Packs all promise data from the runtime stack into the communication buffer (comm_buffer). This
	// could be useful for backing up program states for debugging or to re-start processing from
	// a previous state.
	//
	// For now, this will not be implemented, as it isn't particularly useful until the system's
	// correctness has been verified.
	//
	*/
	 __device__  void pull_promises(Instance &instance){

		/*
		// [TODO] NOT YET IMPLEMENTED
		*/

	}



	/*
	// The workhorse of the program. This function executes until either a halting condition
	// is encountered or a maximum number of processing cycles has occured. This makes sure
	// that long-running programs don't time out on the GPU. In practice, cycle_count may have
	// to be tuned to the average cycle execution time for a given application. This could
	// potentially be automated using an exponential backoff heuristic.
	*/
	 __device__ void exec(unsigned int cycle_count){

		/* Initialize per-warp resources */
		init_group();

		/* Initialize per-thread resources */
		init_thread();

		PROGRAM_SPEC::initialize(*this);

		/*
		if(util::current_leader()){
			printf("\n\n\nInitial frame zero resident count is: %d\n\n\n",_dev_ctx.stack->frames[0].children_residents);
		}
		*/

		/* The execution loop. */
		#ifdef RACE_COND_PRINT
		unsigned int cycle_break = cycle_count;
		#endif
		for(unsigned int cycle=0u; cycle<cycle_count; cycle++){
			/* Early halting handled with a break. */
			exec_cycle();
			if(!_grp_ctx.keep_running){
				#ifdef RACE_COND_PRINT
				cycle_break = cycle+1;
				#endif
				break;
			}
		}

		PROGRAM_SPEC::finalize(*this);

		/*
		// Ensure that nothing which should persist between dispatches is lost in the
		// shared or private memory of the halting program.
		*/
		cleanup_runtime();

		if(util::current_leader()){
			rc_printf("SM %d finished after %d cycles with promise delta %d\n",threadIdx.x,cycle_break,_grp_ctx.SM_promise_delta);
		}

	}





	#if 1


	__host__ static bool queue_count(Instance runtime, LinkType* host_arena, QueueType queue, LinkAdrType& result){

		//printf("Entered function\n");
		LinkAdrType head = queue.get_head();
		LinkAdrType tail = queue.get_tail();
		LinkAdrType last = LinkAdrType::null;
		LinkAdrType count = 0;

		//printf("About to check if the head or tail was NULL\n");

		if( head.is_null() ){
			if( tail.is_null() ) {
				result = 0;
				return true;
			} else {
				printf("NULL head with a non-NULL tail\n");
				return false;
			}
		} else if ( tail.is_null() ){
			printf("Non-NULL head with a NULL tail\n");
			return false;
		}

		//printf("Just checked if the head or tail was NULL\n");
		LinkAdrType iter = head;
		while( ! iter.is_null() ){
			if( iter.adr > runtime.arena_size ){
				printf("Queue has bad index pointing to index %d\n",iter.adr);
				return false;
			}
			if(host_arena[iter.adr].meta_data != 0){
				printf("Link re-visited\n");
				LinkAdrType loop_point = iter;
				LinkAdrType visit_count = 0;
				iter = head;
				printf("(%d,%d): ",head.adr,tail.adr);
				LinkAdrType step_count = 0;
				while(true){
					if(iter == loop_point){
						if(visit_count.adr == 0){
							printf("{%d}->",iter.adr);
						} else {
							printf("{%d}\n",iter.adr);
							break;
						}
						visit_count.adr += 1;
					} else {
						printf("%d->",iter.adr);
					}
					iter = host_arena[iter.adr].next;
					if( iter.is_null() ){
						printf("NULL\n",iter.adr);
						return false;
					}
					step_count.adr +=1;
					if(step_count.adr > 64){
						printf("...{%d}\n",loop_point.adr);
						break;
					}
				}
				return false;
			} else {
				host_arena[iter.adr].meta_data = 1;
			}
			last = iter;
			iter = host_arena[iter.adr].next;
			count.adr += 1;
		}

		if( last.adr != tail.adr ){
			printf("Final link %d in the queue (%d,%d) not the tail\n",last.adr,head.adr,tail.adr);
			return false;
		}

		result = count;
		return true;

	}






	/*
	// Counts the number of links in each queue in the pool and in the stack, storing the counts in
	// the provided arrays. This funciton returns true if counting was successful and returns false
	// if an invalid state is detected.
	*/
	__host__  static bool runtime_overview(Instance runtime){

		bool result = true;

		#ifdef DEBUG_PRINT
		const bool always_print = true;
		#else
		const bool always_print = true;
		#endif

		LinkAdrType* pool_counts  = new LinkAdrType[POOL_SIZE];
		bool*         pool_count_validity  = new bool[POOL_SIZE];
		LinkAdrType* stack_counts = new LinkAdrType[STACK_SIZE*FRAME_SIZE];
		bool*         stack_count_validity = new bool[STACK_SIZE*FRAME_SIZE];


		#ifdef LAZY_LINK
		AdrType claim_count;
		runtime.claim_count >> claim_count;
		#endif

		LinkType* host_arena = new LinkType[runtime.arena_size];

		QueueType* host_pool = new QueueType[POOL_SIZE];

		StackType* host_stack = new StackType;


		LinkAdrType link_total = 0;

		cudaMemcpy(host_arena,runtime.arena,sizeof(LinkType) *runtime.arena_size,cudaMemcpyDeviceToHost);
		cudaMemcpy(host_pool ,runtime.pool ,sizeof(QueueType)*POOL_SIZE ,cudaMemcpyDeviceToHost);
		cudaMemcpy(host_stack,runtime.stack,sizeof(StackType)           ,cudaMemcpyDeviceToHost);


		for(AdrType i = 0; i < runtime.arena_size; i++){
			host_arena[i].meta_data = 0;
		}



		//printf("Counting through pool links...\n");
		for(int i=0; i < POOL_SIZE; i++){
			//printf("Counting pool queue %d\n",i);
			QueueType queue = host_pool[i];
			//printf("Read pool queue %d\n",i);
			pool_count_validity[i] = queue_count(runtime,host_arena,queue,pool_counts[i]);
			//printf("Just validated pool queue %d\n",i);
			result = result && pool_count_validity[i];
			if(pool_count_validity[i]){
				link_total.adr += pool_counts[i].adr;
			}
		}

		//printf("Counting through stack links...\n");
		for(int i=0; i < STACK_SIZE; i++){
			for(int j=0; j < FRAME_SIZE; j++){
				QueueType queue = host_stack->frames[i].pool.queues[j];
				unsigned int index = i*FRAME_SIZE + j;
				stack_count_validity[i] = queue_count(runtime,host_arena,queue,stack_counts[index]);
				result = result && stack_count_validity[i];
				if(stack_count_validity[i]){
					link_total.adr += stack_counts[index].adr;
				}
			}

		}


		if( (!result) || always_print ){
			printf("POOL:\t[");
			for(int i=0; i<POOL_SIZE; i++){
				if(pool_count_validity[i]){
					printf("\t%d",pool_counts[i].adr);
				} else {
					printf("\t????");
				}
			}
			printf("\t]\n");

			unsigned int status_flags	= host_stack->status_flags;
			unsigned int depth	= (host_stack->depth_live >> 16) & 0xFFFF;
			unsigned int live	= (host_stack->depth_live) & 0xFFFF;

			#ifdef LAZY_LINK
			printf("CLAIM_COUNT:\t%d\n",claim_count);
			#endif

			printf("STACK:\t(status_flags: %#010x\tdepth: %d\tlive: %d)\t{\n",status_flags,depth,live);
			for(int i=0; i < STACK_SIZE; i++){
				util::mem::PairPack<PromiseCount> child_res = host_stack->frames[i].children_residents.data;
				PromiseCount children  = child_res.get_left();
				PromiseCount residents = child_res.get_right();
				printf("(children: %d\t residents: %d)\t[",children,residents);
				for(int j=0; j < FRAME_SIZE; j++){
					unsigned int index = i*FRAME_SIZE + j;
					if(stack_count_validity[i]){
						printf("\t%d",stack_counts[index].adr);
					} else {
						printf("\t????");
					}
				}
				printf("\t]\n");

			}
			printf("} LINK TOTAL: %d\n",link_total.adr);
		}

		delete[] host_arena;
		delete[] host_pool;
		delete   host_stack;

		delete[] pool_count_validity;
		delete[] stack_count_validity;

		return result;

	}

	#endif


	template<typename TYPE,typename... ARGS>
	__device__ void async(ARGS... args){
		async_call_cast<TYPE>(0,args...);
	}




	template<typename TYPE,typename... ARGS>
	__device__  void sync(ARGS... args){
		immediate_call_cast<TYPE>(Promise<TYPE>(args...));
	}


	template<typename TYPE>
	__device__ float load_fraction()
	{
		return NAN;
	}




};




#undef MEMBER_SWITCH
#undef CONST_SWITCH

#endif


