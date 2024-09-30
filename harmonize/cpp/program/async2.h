









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
	struct Lookup { typedef typename PromiseUnionType::template Lookup<TYPE>::type type; };

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
		adapt::GPUrt::barrier<adapt::GPUrt::thread_scope_system> load_barrier;
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



}















