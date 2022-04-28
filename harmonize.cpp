



#define HARMONIZE

//#define DEBUG_PRINT
//#define RACE_COND_PRINT
//#define QUEUE_PRINT

#define INF_LOOP_SAFE


#define NOOP(x) ;

#ifdef QUEUE_PRINT
	#define q_printf  printf
#else
	#define q_printf(fmt, ...) NOOP(...);
#endif


#ifdef RACE_COND_PRINT
	#define rc_printf  printf
#else
	#define rc_printf(fmt, ...) ;
#endif


#ifdef DEBUG_PRINT
	#define db_printf  printf
#else
	#define db_printf(fmt, ...) ;
#endif


//#define HRM_TIME 16

#ifdef HRM_TIME
	#define beg_time(idx) if(util::current_leader()) { _grp_ctx.time_totals[idx] -= clock64(); }
	#define end_time(idx) if(util::current_leader()) { _grp_ctx.time_totals[idx] += clock64(); }
#else
	#define beg_time(idx) ;
	#define end_time(idx) ;
#endif

#include "util/util.cpp"



template <typename... TAIL>
struct ArgTuple;

template <>
struct ArgTuple <>
{
	__host__ __device__ ArgTuple<>(){}
};

template <typename HEAD, typename... TAIL>
struct ArgTuple <HEAD, TAIL...>
{
	typedef ArgTuple<TAIL...> Tail;

	HEAD head;
	Tail tail;
	
	__host__ __device__ ArgTuple<HEAD,TAIL...> (HEAD h, TAIL... t)
		: head(h)
		, tail(t...)

	{}
};


template<typename TYPE>
struct OpType;

template<typename RETURN, typename... ARGS>
struct OpType < RETURN (*) (ARGS...) > {
	typedef RETURN Return;
	typedef ArgTuple<ARGS...> Args;
};




template<typename OPERATION>
struct Promise
{

	typedef typename OpType< typename OPERATION::Type >::Return Return;
	typedef typename OpType< typename OPERATION::Type >::Args   Args;

	Args args;

	template<typename PROGRAM, typename... TUPLE_ARGS, typename... UNROLL_ARGS>
	__device__ Return inner_eval(PROGRAM program, ArgTuple<TUPLE_ARGS...> tuple_args, UNROLL_ARGS... unroll_args)
	{
		return inner_eval(program,tuple_args.tail,unroll_args...,tuple_args.head);
	}


	template<typename PROGRAM, typename... UNROLL_ARGS>
	__device__ Return inner_eval(PROGRAM program, ArgTuple<> empty_tuple_args, UNROLL_ARGS... unroll_args)
	{
		return OPERATION::template eval<PROGRAM>(program,unroll_args...);
	}


	template<typename PROGRAM>
	__device__ Return operator() (PROGRAM program) {
		return inner_eval(program,args);
	}


	template<typename... ARGS>
	__host__ __device__ Promise<OPERATION> ( ARGS... a ) : args(a...) {}	

};



//!
//! The `ProgramStateDef` template struct is used to define the state that is stored in each scope
//! of a program class.
//!
template <typename DEVICE_STATE, typename GROUP_STATE, typename THREAD_STATE>
struct ProgramStateDef
{
	typedef DEVICE_STATE DeviceState;
	typedef GROUP_STATE  GroupState;
	typedef THREAD_STATE ThreadState;
};




using OpDisc = unsigned int;

template<typename... TYPES>
struct OpUnion {};


template <typename... TYPES>
struct OpUnionLookup;

template <typename QUERY>
struct OpUnionLookup <QUERY> {
	static const bool   CONTAINED = false;
	static const OpDisc LEVEL     = 0;
	static const OpDisc Q_LEV     = LEVEL;
	static const OpDisc DISC      = LEVEL;
};

template <typename QUERY, typename HEAD, typename... TAIL>
struct OpUnionLookup <QUERY, HEAD, TAIL...> {
	static const bool   MATCHES   = std::is_same<QUERY,HEAD>::value ;
	static const bool   CONTAINED = MATCHES || (OpUnionLookup<QUERY,TAIL...>::CONTAINED);
	static const OpDisc LEVEL     = OpUnionLookup<QUERY,TAIL...>::LEVEL + 1;
	static const OpDisc Q_LEV     = MATCHES ? LEVEL : (OpUnionLookup<QUERY,TAIL...>::Q_LEV);
	static const OpDisc DISC      = LEVEL - Q_LEV;
};


template <typename OP_UNION>
union PromiseUnion;


struct VoidState {};


//!
//! The base case of the `PromiseUnion` template union defines empty functions to cap off the
//! recursion of non-base cases when evaluating promises. 
//!
template <>
union PromiseUnion <OpUnion<>> {

	static const OpDisc COUNT = 0;


	template <typename PROGRAM, typename TYPE>
	__host__  __device__ void rigid_eval( PROGRAM program ) {
		return;
	}

	template <typename PROGRAM>
	__host__  __device__ void loose_eval( PROGRAM program, OpDisc op_disc ) {
		return;
	}

	__host__ __device__ void dyn_copy_as( OpDisc op_disc, PromiseUnion<OpUnion<>>& other){ }

	PromiseUnion<OpUnion<>> () = default;

};



//!
//! The recursive case of the `PromiseUnion` template union defines the `cast()`, `rigid_eval()`,
//! and `loose_eval()` template functions for the async function/parameter type corresponding to
//! the first template argument.
//!
template <typename HEAD, typename... TAIL>
union PromiseUnion<OpUnion<HEAD, TAIL...>>
{

	typedef Promise<HEAD> Head;
	typedef PromiseUnion<OpUnion<TAIL...>> Tail;
	

	Head head_form;
	Tail tail_form;

	public:

	static const OpDisc COUNT = sizeof...(TAIL) + 1;
	static const OpDisc INDEX = OpUnionLookup<HEAD,HEAD,TAIL...>::DISC;

	template <typename TYPE>
	struct Lookup { typedef OpUnionLookup<TYPE,HEAD,TAIL...> type; };


	template <typename TYPE>
	__host__  __device__ typename std::enable_if<
		std::is_same<TYPE,HEAD>::value, 
		Promise<TYPE>&
	>::type
	cast() {
		return head_form;
	}

	template <typename TYPE>
	__host__  __device__ typename std::enable_if<
		(!std::is_same<TYPE,HEAD>::value) && OpUnionLookup<TYPE,TAIL...>::CONTAINED,
		Promise<TYPE>&
	>::type
	cast(){	
		return tail_form.template cast<TYPE>();
	}

	template <typename TYPE>
	__host__  __device__ typename std::enable_if<
		(!std::is_same<TYPE,HEAD>::value) && (!OpUnionLookup<TYPE,TAIL...>::CONTAINED),
		Promise<TYPE>&
	>::type
	cast (){	
		static_assert( (!OpUnionLookup<TYPE,TAIL...>::CONTAINED), "Promise type does not exist in union" );
	}


	template <typename PROGRAM, typename TYPE >
	__host__  __device__ typename std::enable_if<
		std::is_same<TYPE,HEAD>::value, 
		void
	>::type
	rigid_eval(
		PROGRAM program
	) {
		head_form(program);
	}

	template <typename PROGRAM, typename TYPE>
	__host__  __device__ typename std::enable_if<
		(!std::is_same<TYPE,HEAD>::value) && OpUnionLookup<TYPE,TAIL...>::CONTAINED,
		void
	>::type
	rigid_eval(
		PROGRAM program
	) {
		tail_form.template rigid_eval<PROGRAM,TYPE>(program);
	}

	template <typename PROGRAM, typename TYPE>
	__host__  __device__ typename std::enable_if<
		(!std::is_same<TYPE,HEAD>::value) && (!OpUnionLookup<TYPE,TAIL...>::CONTAINED),
		void
	>::type
	rigid_eval(
		PROGRAM program
	) {
		static_assert( (!OpUnionLookup<TYPE,TAIL...>::CONTAINED), "Promise type does not exist in union" );
	}


	template <typename PROGRAM>
	__host__  __device__ void loose_eval (
		PROGRAM program,
		OpDisc disc
	) {
		if(disc == INDEX){
			head_form(program);
		} else {
			tail_form. template loose_eval<PROGRAM>(program,disc-1);
		}

	}

	template <typename TYPE>
	__host__ __device__ void copy_as(PromiseUnion<OpUnion<HEAD,TAIL...>>& other){
		cast<TYPE>() = other.template cast<TYPE>();
	}


	__host__ __device__ void dyn_copy_as(OpDisc disc, PromiseUnion<OpUnion<HEAD,TAIL...>>& other){
		if( disc == INDEX ){
			cast<HEAD>() = other.cast<HEAD>();
		} else {
			tail_form.dyn_copy_as(disc-1,other.tail_form);
		}
	}

	__host__ __device__ PromiseUnion<OpUnion<HEAD,TAIL...>> () : tail_form() {}

};






template <typename OP_UNION>
struct PromiseEnum {

	PromiseUnion<OP_UNION> data;
	OpDisc                 disc;
	
	PromiseEnum() = default;

	__host__ __device__ PromiseEnum(PromiseUnion<OP_UNION> uni, OpDisc d)
		: data(uni)
		, disc(d)
	{ }

};



//!
//! The `WorkLink` template struct, given a `PromiseUnion` union, an address type, and a group
//! size, stores an array of `GROUP_SIZE` promise unions of the corresponding type and an
//! address value of type `ADR_TYPE`. Instances of this template also contain a `Op` value to
//! identify what type of work is contained within the link, a `meta_data` field, and a `count`
//! field to indicate the number of contained promises.
//!
template <typename PROMISE_UNION, typename ADR_TYPE, size_t GROUP_SIZE>
struct WorkLink
{

	typedef ADR_TYPE AdrType;
	typedef PROMISE_UNION PromiseType;

	PromiseType    promises[GROUP_SIZE];

	AdrType        next;
	unsigned int   meta_data;
	OpDisc         id;
	unsigned short count;


	/*
	// Zeros out a link, giving it a promise count of zero, a null function ID, and sets next
	// to the given input.
	*/
	__host__ __device__ void empty(AdrType next_adr){

		next	= next_adr;
		id	= PromiseType::COUNT;
		count	= 0;

	}


};




template <typename PROMISE_UNION, typename ADR_TYPE>
struct PromiseLink
{
	
	typedef ADR_TYPE                 AdrType;
	typedef PROMISE_UNION            PromiseType;
	typedef PromiseEnum<PromiseType> PromiseEnumType;

	PromiseEnumType data;
	AdrType         next;

	__host__ __device__ void empty(AdrType next_adr, PromiseEnumType value){
		next	= next_adr;
		data	= value;
	}

};


template <typename PROMISE_UNION, typename ADR_TYPE, typename COUNTER_TYPE=unsigned int>
struct WorkBarrier {
		
	typedef ADR_TYPE                 AdrType;
	typedef PROMISE_UNION            PromiseType;
	typedef PromiseEnum<PromiseType> PromiseEnumType;
	typedef COUNTER_TYPE             CounterType;

	CounterType counter;
	AdrType     work;

	WorkBarrier<PromiseType,AdrType,CounterType>() = default;
	
	__host__ __device__ WorkBarrier<PromiseType,AdrType,CounterType>(
		CounterType dependency_count
	) {
		counter = dependency_count;
	}

	__device__ void await (PromiseEnumType promise) {

	}


};





//!
//! The `WorkArena` template struct accepts an address type and a `WorkLink` struct type and
//! represents a buffer of work links of the given class indexable by the given address type.
//!
template <typename ADR_TYPE, typename LINK_TYPE>
struct WorkArena
{

	typedef LINK_TYPE LinkType;
	typedef ADR_TYPE  LinkAdrType;

	static const size_t max_size = ADR_TYPE::null;
	size_t     size;
	LinkType *links;

	__host__ __device__ LinkType& operator[](LinkAdrType adr){
		return links[adr.adr];
	}

};




//!
//! The `WorkPool` template struct accepts a queue type and a `size_t` count and represents an
//! array of `QUEUE_COUNT` queues.
//!
template <typename QUEUE_TYPE, size_t QUEUE_COUNT>
struct WorkPool
{

	static const size_t size = QUEUE_COUNT;
	QUEUE_TYPE queues[QUEUE_COUNT];
	
};




template <typename QUEUE_TYPE, size_t QUEUE_COUNT>
struct WorkFrame
{

	unsigned int children_residents;
	WorkPool<QUEUE_TYPE, QUEUE_COUNT> pool;

};




template<typename FRAME_TYPE, size_t STACK_SIZE = 0>
struct WorkStack {
	static const bool   FLAT       = false;

	static const size_t PART_MULT  = 3;

	static const size_t NULL_LEVEL = STACK_SIZE;
	static const size_t MAX_LEVEL  = STACK_SIZE-1;

	unsigned int    checkout;
	unsigned int	status_flags;
	unsigned int	depth_live;
	FRAME_TYPE frames[STACK_SIZE];

};

template<typename FRAME_TYPE>
struct WorkStack<FRAME_TYPE, 0>
{
	static const bool   FLAT       = true;
	
	static const size_t PART_MULT  = 1;
	static const size_t NULL_LEVEL = 1;
	
	unsigned int    checkout;
	unsigned int	status_flags;
	unsigned int	depth_live;
	FRAME_TYPE frames[1];

};



namespace detector {

	template <class... TYPES>
	using void_t = void;

	template <size_t VALUE>
	using void_size = void;

	template <template <class...> class LOOKUP, class GUARD, class... ARGS>
	struct is_detected : std::false_type{};

	template <template <class...> class LOOKUP, class... ARGS>
	struct is_detected<LOOKUP, void_t<LOOKUP<ARGS...>>, ARGS...> : std::true_type{};

	template<bool COND, class TRUE_TYPE = void, class FALSE_TYPE = void>
	struct type_switch
	{};

	template<class TRUE_TYPE, class FALSE_TYPE>
	struct type_switch<true, TRUE_TYPE,FALSE_TYPE>
	{
		typedef TRUE_TYPE type;
	};

	template<class TRUE_TYPE, class FALSE_TYPE>
	struct type_switch<false, TRUE_TYPE, FALSE_TYPE>
	{
		typedef FALSE_TYPE type;
	};


}

template<template<class...> class LOOKUP, class... ARGS>
using is_detected = typename detector::is_detected<LOOKUP,void,ARGS...>::type;

template<class DEFAULT, template<class> class LOOKUP, class TYPE>
using type_switch = typename detector::type_switch<is_detected<LOOKUP,TYPE>::value ,TYPE,DEFAULT>::type;


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



#define LAZY_LINK




template< typename PROGRAM_SPEC >
class HarmonizeProgram
{

	public:

	typedef HarmonizeProgram<PROGRAM_SPEC> ProgramType;

	#define MEMBER_GUARD(NAME,DEFAULT) \
		struct NAME##Default { typedef unsigned int NAME; }; \
		template<class TYPE> using  NAME##Check = typename TYPE::NAME; \
		typedef typename type_switch<NAME##Default,NAME##Check,PROGRAM_SPEC>::NAME NAME;

	/*
	struct AdrTypeDefault { typedef unsigned int AdrType; };
	template<class TYPE> using  AdrTypeCheck = typename TYPE::AdrType;
	typedef typename type_or<AdrTypeDefault,AdrTypeCheck,PROGRAM_SPEC>::AdrType AdrType;
	*/

	MEMBER_GUARD(    AdrType,unsigned int)
	MEMBER_GUARD(      OpSet,   OpUnion<>)
	MEMBER_GUARD(DeviceState,   VoidState)
	MEMBER_GUARD( GroupState,   VoidState)
	MEMBER_GUARD(ThreadState,   VoidState)

	#undef MEMBER_GUARD

	typedef PromiseUnion<OpSet> PromiseUnionType;

	template<typename TYPE>
	struct Lookup { typedef typename PromiseUnionType::Lookup<TYPE>::type type; };

	#define SIZE_T_GUARD(NAME,DEFAULT) \
		struct NAME##Default { static const size_t NAME = DEFAULT; }; \
		template<class TYPE> using  NAME##Check = decltype( TYPE::NAME ); \
		static const size_t NAME = type_switch<NAME##Default,NAME##Check,PROGRAM_SPEC>::NAME;
	
	SIZE_T_GUARD(STASH_SIZE,16)
	SIZE_T_GUARD(FRAME_SIZE,32)
	SIZE_T_GUARD( POOL_SIZE,32)
	SIZE_T_GUARD(GROUP_SIZE,32)
	SIZE_T_GUARD(STACK_SIZE, 0)


	#undef SIZE_T_GUARD


	/*
	// The number of async functions present in the program.
	*/
	static const unsigned char FN_ID_COUNT = PromiseUnionType::COUNT;


	/*
	// During system verification/debugging, this will be used as a cutoff to prevent infinite
	// looping
	*/
	static const unsigned int PUSH_QUEUE_RETRY_LIMIT         = 32;
	static const unsigned int FILL_STASH_RETRY_LIMIT         = 1;
	static const unsigned int FILL_STASH_LINKS_RETRY_LIMIT   = 32;
	

	static const size_t       WORK_GROUP_SIZE  = GROUP_SIZE;

	/*
	// A set of halting condition flags
	*/
	static const unsigned int BAD_FUNC_ID_FLAG	= 0x00000001;
	static const unsigned int STASH_FAIL_FLAG	= 0x00000002;
	static const unsigned int COMPLETION_FLAG	= 0x80000000;
	static const unsigned int EARLY_HALT_FLAG	= 0x40000000;


	typedef util::mem::Adr       <AdrType>             LinkAdrType;
	typedef util::mem::PoolQueue <LinkAdrType>         QueueType;
	typedef WorkFrame       <QueueType,FRAME_SIZE>     FrameType;
	typedef WorkStack       <FrameType,STACK_SIZE>     StackType;
	typedef WorkPool        <QueueType,POOL_SIZE>      PoolType;

	typedef WorkLink        <PromiseUnionType, LinkAdrType, WORK_GROUP_SIZE> LinkType;
	
	typedef WorkArena       <LinkAdrType,LinkType>     ArenaType;



	/*
	// The depth of the partial table (1 if stack is flat, 3 otherwise).
	*/
	static const unsigned char PART_TABLE_DEPTH = StackType::PART_MULT;
	static const unsigned char PART_ENTRY_COUNT = FN_ID_COUNT*PART_TABLE_DEPTH;


	
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

		size_t				level;		// Current level being run
		
		bool				keep_running;
		bool				busy;
		bool				can_make_work;	
		bool				scarce_work;	

		unsigned char			stash_count;	// Number of filled blocks in stash
		unsigned char			exec_head;	// Indexes the link that is/will be evaluated next
		unsigned char			full_head;	// Head of the linked list of full links
		unsigned char			empty_head;	// Head of the linked list of empty links

		unsigned char			partial_map[FN_ID_COUNT*PART_TABLE_DEPTH]; // Points to partial blocks

		unsigned char			link_stash_count; // Number of device-space links stored

		LinkType			stash[STASH_SIZE];
		LinkAdrType			link_stash[STASH_SIZE];
		
		
		int				SM_promise_delta;
		unsigned long long int		work_iterator;

		#ifdef HRM_TIME
		unsigned long long int		time_totals[HRM_TIME];
		#endif

	
	};


	/*
	// This struct represents the entire set of data structures that must be stored in device
	// memory to track the state of the program defined by the developer as well as the state
	// of the context which is driving execution.
	*/
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

	DeviceContext & _dev_ctx;
	GroupContext  & _grp_ctx;
	ThreadContext & _thd_ctx;

	public:

	DeviceState   &   device;
	GroupState    &    group;
	ThreadState   &   thread;

	__device__ HarmonizeProgram<PROGRAM_SPEC> (
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






	/*
	// Instances wrap around their program scope's DeviceContext. These differ from a program's
	// DeviceContext object in that they perform automatic deallocation as soon as they drop
	// out of scope.
	*/
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
			return (base_cr != 0);
		}

	};





	protected:
	
	/*
	// Returns an index into the partial map of a group based off of a function id and a depth. If
	// an invalid depth or function id is used, PART_ENTRY_COUNT is returned.
	*/
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


	/*
	// Initializes the shared state of a work group, which is stored as a ctx_shared struct. This
	// is mainly done by initializing handles to the arena, pool, and stack, setting the current
	// level to null, setting the stash iterator to null, and zeroing the stash.
	*/
	 __device__  void init_group(){

		unsigned int active = __activemask();

		__syncwarp(active);

		if(util::current_leader()){

			if( StackType::FLAT ){
				_grp_ctx.level = 0;
			} else {
				_grp_ctx.level = StackType::NULL_LEVEL;
			}

			_grp_ctx.stash_count = 0;
			_grp_ctx.link_stash_count = 0;
			_grp_ctx.keep_running = true;
			_grp_ctx.busy 	 = false;
			_grp_ctx.can_make_work= true;
			_grp_ctx.exec_head    = STASH_SIZE;
			_grp_ctx.full_head    = STASH_SIZE;
			_grp_ctx.empty_head   = 0;
			_grp_ctx.work_iterator= 0;
			_grp_ctx.scarce_work  = false;

			for(unsigned int i=0; i<STASH_SIZE; i++){
				_grp_ctx.stash[i].empty(i+1);
			}
				
			for(unsigned int i=0; i<PART_ENTRY_COUNT; i++){
				_grp_ctx.partial_map[i] = STASH_SIZE;
			}

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

	/*
	// Returns the current highest level in the stack. Given that this program is highly parallel,
	// this number inherently cannot be trusted. By the time the value is fetched, the stack could
	// have a different height or the thread that set the height may not have deposited links in the
	// corresponding level yet.
	*/
	 __device__  unsigned int highest_level(){

		return left_half(_dev_ctx.stack->depth_live);

	}


	/*
	// Returns a reference to the frame at the requested level in the stack.
	*/
	 __device__  FrameType& get_frame(unsigned int level){

		return _dev_ctx.stack->frames[level];

	}


	/*
	// Joins two queues such that the right queue is now at the end of the left queue.
	//
	// WARNING: NOT THREAD SAFE. Only use this on queues that have been claimed from the stack
	// atomically. If not, one of the queues manipulated with this function will almost certainly
	// become malformed at some point. Woe betide those that do not heed this dire message.
	*/
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

			/*
			// Find last link in the queue referenced by left_queue.
			*/
			LinkType& left_tail = _dev_ctx.arena[left_tail_adr];

			/*
			// Set the index for the tail's successor to the head of the queue referenced by
			// right_queue.
			*/
			left_tail.next = right_head_adr;

			/* Set the right half of the left_queue handle to index the new tail. */
			left_queue.set_tail(right_tail_adr);
			
			result = left_queue;
			
		}
		
		return result;

	}


	/*
	// Takes the first link off of the queue and returns the index of the link in the arena. If the
	// queue is empty, a null address is returned instead.
	//
	// WARNING: NOT THREAD SAFE. Only use this on queues that have been claimed from the stack
	// atomically. If not, one of the queues manipulated with this function will almost certainly
	// become malformed at some point. Woe betide those that do not heed this dire message.
	*/
	 __device__  LinkAdrType pop_front(QueueType& queue){

		LinkAdrType result;
		/*
		// Don't try unless the queue is non-null
		*/
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


	/*
	// Adds the given link to the end of the given queue. This link can NOT be part of another queue,
	// and its next pointer will be automatically nulled before it is appended. If you need to merge
	// two queues together, use join_queues.
	//
	// WARNING: NOT THREAD SAFE. Only use this on queues that have been claimed from the stack
	// atomically. If not, one of the queues manipulated with this function will almost certainly
	// become malformed at some point. Woe betide those that do not heed this dire message.
	*/
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



	/*
	// Attempts to pull a queue from a range of queue slots, trying each slot starting from the given
	// starting index onto the end of the range and then looping back from the beginning. If, after
	// trying every slot in the range, no non-null queue was obtained, a QueueType::null value is returned.
	*/
	 __device__  QueueType pull_queue(QueueType* src, unsigned int start_index, unsigned int range_size, unsigned int& src_index){

		QueueType result;
		
		__threadfence();
		/*
		// First iterate from the starting index to the end of the queue range, attempting to
		// claim a non-null queue until either there are no more slots to try, or the atomic
		// swap successfuly retrieves something.
		*/
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

		/*
		// Continue searching from the beginning of the range to just before the beginning of the
		// previous scan.
		*/
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
		/*
		// Return QueueType::null if nothing is found
		*/
		result.pair.data = QueueType::null;
		return result;


	}




	/*
	// Repeatedly tries to push a queue to a destination queue slot by atomic exchanges. If a non
	// null queue is ever returned by the exchange, it attempts to merge with a subsequent exchange.
	// For now, until correctness is checked, this process repeats a limited number of times. In
	// production, this will be an infinite loop, as the function should not fail if correctly 
	// implemented.
	*/
	 __device__  void push_queue(QueueType& dest, QueueType queue){

		if( queue.is_null() ){
			return;
		}
		#ifdef INF_LOOP_SAFE
		while(true)
		#else
		for(int i=0; i<PUSH_QUEUE_RETRY_LIMIT; i++)
		#endif
		{
			__threadfence();
			
			QueueType swap;
			swap.pair.data = atomicExch(&dest.pair.data,queue.pair.data);
			/*
			// If our swap returns a non-null queue, we are still stuck with a queue that
			// needs to be offloaded to the stack. In this case, claim the queue from the 
			// slot just swapped with, merge the two, and attempt again to place the queue
			// back. With this method, swap failures are bounded by the number of pushes to
			// the queue slot, with at most one failure per push_queue call, but no guarantee
			// of which attempt from which call will suffer from an incurred failure.
			*/
			if( ! swap.is_null() ){
				q_printf("Ugh. We got queue (%d,%d) when trying to push a queue\n",swap.get_head().adr,swap.get_tail().adr);
				QueueType other_swap;
				other_swap.pair.data = atomicExch(&dest.pair.data,QueueType::null); 
				queue = join_queues(other_swap,swap);
				q_printf("Merged it to form queue (%d,%d)\n",queue.get_head().adr,queue.get_tail().adr);
			} else {
				q_printf("Finally pushed (%d,%d)\n",queue.get_head().adr,queue.get_tail().adr);
				break;
			}
		}


	}




	/*
	// Claims a link from the link stash. If no link exists in the stash, LinkAdrType::null is returned.
	*/
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



	/*
	// Inserts an empty slot into the stash. This should only be called if there is enough space in
	// the link stash.
	*/
	 __device__  void insert_stash_link(LinkAdrType link){

		unsigned int count = _grp_ctx.link_stash_count;
		_grp_ctx.link_stash[count] = link;
		_grp_ctx.link_stash_count = count + 1;
		q_printf("New link stash count: %d\n",_grp_ctx.link_stash_count);

	}




	/*
	// Claims an empty slot from the stash and returns its index. If no empty slot exists in the stash,
	// then STASH_SIZE is returned.
	*/
	 __device__  unsigned int claim_empty_slot(){

		unsigned int slot = _grp_ctx.empty_head;
		if(slot != STASH_SIZE){
			_grp_ctx.empty_head = _grp_ctx.stash[slot].next.adr;
			db_printf("EMPTY: %d << %d\n",slot,_grp_ctx.empty_head);
		}
		return slot;

	}


	/*
	// Inserts an empty slot into the stash. This should only be called if there is enough space in
	// the link stash.
	*/
	 __device__  void insert_empty_slot(unsigned int slot){

		_grp_ctx.stash[slot].next.adr = _grp_ctx.empty_head;
		db_printf("EMPTY: >> %d -> %d\n",slot,_grp_ctx.empty_head);
		_grp_ctx.empty_head = slot;

	}


	/*
	// Claims a full slot from the stash and returns its index. If no empty slot exists in the stash,
	// then STASH_SIZE is returned.
	*/
	 __device__  unsigned int claim_full_slot(){

		unsigned int slot = _grp_ctx.full_head;
		if(slot != STASH_SIZE){
			_grp_ctx.full_head = _grp_ctx.stash[slot].next.adr;
			db_printf("FULL : %d << %d\n",slot,_grp_ctx.full_head);
		}
		return slot;

	}


	/*
	// Inserts a full slot into the stash. This should only be called if there is enough space in
	// the link stash.
	*/
	 __device__  void insert_full_slot(unsigned int slot){

		_grp_ctx.stash[slot].next.adr = _grp_ctx.full_head;
		db_printf("FULL : >> %d -> %d\n",slot,_grp_ctx.full_head);
		_grp_ctx.full_head = slot;

	}




	/*
	// Attempts to fill the link stash to the given threshold with links. This should only ever
	// be called in a single-threaded manner.
	*/
	 __device__  void fill_stash_links(unsigned int threashold){



		unsigned int active = __activemask();
		__syncwarp(active);

		if( util::current_leader() ){

			#ifdef LAZY_LINK

			#if 1
			unsigned int wanted = threashold - _grp_ctx.link_stash_count; 
			if( (_grp_ctx.link_stash_count < threashold) && ((*_dev_ctx.claim_count) <  _dev_ctx.arena.size) ){
				AdrType claim_offset = atomicAdd(_dev_ctx.claim_count,wanted);
				unsigned int received = 0;
				if( claim_offset <= (_dev_ctx.arena.size - wanted) ){
					received = wanted;
				} else if ( claim_offset >= _dev_ctx.arena.size ) {
					received = 0;
				} else {
					received = _dev_ctx.arena.size - claim_offset;
				}
				for( unsigned int index=0; index < received; index++){
					insert_stash_link(LinkAdrType(claim_offset+index));
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
		
				if(_grp_ctx.link_stash_count >= threashold){
					break;
				}
				/*	
				// Attempt to pull a queue from the pool. This should be very unlikely to fail unless
				// almost all links have been exhausted or the pool size is disproportionately small
				// relative to the number of work groups. In the worst case, this should simply not 
				// al_thd_ctxate any links, and the return value shall report this.
				*/
				unsigned int src_index = LinkAdrType::null;
				unsigned int start = util::random_uint(_thd_ctx.rand_state)%POOL_SIZE;
				QueueType queue = pull_queue(_dev_ctx.pool->queues,start,POOL_SIZE,src_index);
				q_printf("Pulled queue (%d,%d) from pool %d\n",queue.get_head().adr,queue.get_tail().adr,src_index);

				/*
				// Keep popping links from the queue until the full number of links have been added or
				// the queue runs out of links.
				*/
				for(int i=_grp_ctx.link_stash_count; i < threashold; i++){
					LinkAdrType link = pop_front(queue);
					if( ! link.is_null() ){
						insert_stash_link(link);
						q_printf("Inserted link %d into link stash\n",link.adr);
					} else {
						break;
					}
				}
				push_queue(_dev_ctx.pool->queues[src_index],queue);
				q_printf("Pushed queue (%d,%d) to pool %d\n",queue.get_head().adr,queue.get_tail().adr,src_index);

			}

		}
		__syncwarp(active);

	}




	/*
	// If the number of links in the link stash exceeds the given threshold value, this function frees
	// enough links to bring the number of links down to the threshold. This should only ever be
	// called in a single_threaded manner.
	*/
	 __device__  void spill_stash_links(unsigned int threashold){

		/*
		// Do not even try if no links can be or need to be removed
		*/
		if(threashold >= _grp_ctx.link_stash_count){
			q_printf("Nothing to spill...\n");
			return;
		}

		/*
		// Find where in the link stash to begin removing links
		*/

		QueueType queue;
		queue.pair.data = QueueType::null;

		/*
		// Connect all links into a queue
		*/
		unsigned int spill_count = _grp_ctx.link_stash_count - threashold;
		for(unsigned int i=0; i < spill_count; i++){
			
			LinkAdrType link = claim_stash_link();
			q_printf("Claimed link %d from link stash\n",link.adr);
			push_back(queue,link);

		}

		_grp_ctx.link_stash_count = threashold;


		/*
		// Push out the queue to the pool
		*/
		q_printf("Pushing queue (%d,%d) to pool\n",queue.get_head().adr,queue.get_tail().adr);
		unsigned int dest_idx = util::random_uint(_thd_ctx.rand_state) % POOL_SIZE;
		push_queue(_dev_ctx.pool->queues[dest_idx],queue);
		
		q_printf("Pushed queue (%d,%d) to pool\n",queue.get_head().adr,queue.get_tail().adr);

	}




	/*
	// Decrements the child and resident counter of each frame corresponding to a call at level
	// start_level in the stack returning to a continuation at level end_level in the stack. To reduce
	// overall contention, decrementations are first pooled through a shared atomic operation before
	// being applied to the stack.
	//
	// A call without a continuation should use this function with start_level == end_level, which
	// simply decrements the resident counter at the call's frame.
	*/
	 __device__  void pop_frame_counters(unsigned int start_level, unsigned int end_level){


		unsigned int depth_dec = 0;
		unsigned int delta;
		unsigned int result;

		FrameType& frame = _dev_ctx.stack->frames[start_level];

		/*
		// Decrement the residents counter for the start level
		*/
		delta = util::active_count();
		if(util::current_leader()){
			result = atomicSub(&frame.children_residents,delta);
			if(result == 0u){
				depth_dec += 1;
			}
		}	

		/*
		// Decrement the children counter for the remaining levels
		*/
		for(int d=(start_level-1); d >= end_level; d--){
			FrameType& frame = _dev_ctx.stack->frames[d];
			delta = util::active_count();
			if(util::current_leader()){
				result = atomicSub(&frame.children_residents,delta);
				if(result == 0u){
					depth_dec += 1;
				}
			}
		}

		/*
		// Update the stack base once all other counters have been updated.
		*/
		if(util::current_leader()){
			result = atomicSub(&(_dev_ctx.stack->depth_live),depth_dec);
			if(result == 0){
				set_flags(_grp_ctx,COMPLETION_FLAG);
			}
		}

	}


	/*
	// Repetitively tries to merge the given queue of promises with the queue at the given index in the
	// frame at the given level on the stack. This function currently aborts if an error flag is set
	// or if too many merge failures occur, however, once its correctness is verified, this function
	// will run forever until the merge is successful, as success is essentially guaranteed by
	// the nature of the process.
	*/
	 __device__  void push_promises(unsigned int level, unsigned int index, QueueType queue, int promise_delta) {


		LinkAdrType tail = queue.get_tail();
		LinkAdrType head = queue.get_head();
		rc_printf("SM %d: push_promises(level:%d,index:%d,queue:(%d,%d),delta:%d)\n",threadIdx.x,level,index,tail.adr,head.adr,promise_delta);
		/*
		// Do not bother pushing a null queue if there is no delta to report
		*/
		if( ( ! queue.is_null() ) || (promise_delta != 0) ){

			/*
			// Change the resident counter of the destination frame by the number of promises
			// that have been added to or removed from the given queue
			*/
			FrameType &dest = get_frame(level);		
			unsigned int old_count;
			unsigned int new_count;
			if(promise_delta >= 0) {
				old_count = atomicAdd(&dest.children_residents,(unsigned int) promise_delta);
				new_count = old_count + (unsigned int) promise_delta;
				if(old_count > new_count){
					rc_printf("\n\nOVERFLOW\n\n");
				}
			} else {
				unsigned int neg_delta = - promise_delta;
				old_count = atomicSub(&dest.children_residents,neg_delta);
				new_count = old_count - neg_delta; 
				if(old_count < new_count){
					rc_printf("\n\nUNDERFLOW\n\n");
				}
			}


			_grp_ctx.SM_promise_delta += promise_delta;
			rc_printf("SM %d-%d: Old count: %d, New count: %d\n",blockIdx.x,threadIdx.x,old_count,new_count);

			
			//rc_printf("SM %d-%d: frame zero resident count is: %d\n",blockIdx.x,threadIdx.x,_dev_ctx.stack->frames[0].children_residents);
			/*
			// If the addition caused a frame to change from empty to non-empty or vice-versa,
			// make an appropriate incrementation or decrementation at the stack base.
			*/
			if( (old_count == 0) && (new_count != 0) ){
				atomicAdd(&(_dev_ctx.stack->depth_live),0x00010000u);
			} else if( (old_count != 0) && (new_count == 0) ){
				atomicSub(&(_dev_ctx.stack->depth_live),0x00010000u);
			} else {
				rc_printf("SM %d: No change!\n",threadIdx.x);
			}

			/*
			// Finally, push the queues
			*/
			push_queue(dest.pool.queues[index],queue);
			rc_printf("SM %d: Pushed queue (%d,%d) to stack at index %d\n",threadIdx.x,queue.get_head().adr,queue.get_tail().adr,index);
		
			if( (_dev_ctx.stack->frames[0].children_residents % 0x1000000 ) == 0 ) {
				//printf("SM %d-%d: After queue pushed to stack, frame zero resident count is: %d\n",blockIdx.x,threadIdx.x,_dev_ctx.stack->frames[0].children_residents);
			}

		}
		rc_printf("(%d) the delta: %d\n",threadIdx.x,promise_delta);

	}



	/*
	// Attempts to pull a queue of promises from the frame in the stack of the given level, starting the 
	// pull attempt at the given index in the frame. If no queue could be pulled after attempting a 
	// pull at each queue in the given frame, a QueueType::null value is returned.
	*/
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


	/*
	// Attempts to pull a queue from any frame in the stack, starting from the highest and working
	// its way down. If no queue could be pulled, a QueueType::null value is returned.
	*/
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





	/*
	// Adds the contents of the stash slot at the given index to a link and returns the index of the 
	// link in the arena. This should only ever be called if there is both a link available to store
	// the data and if the index is pointing at a non-empty slot. This also should only ever be
	// called in a single-threaded context.
	*/
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
			_grp_ctx.stash_count -= 1;
		//}

		
		//__syncwarp(active);
		return result;

	}








	/*
	// Removes all promises in the stash that do not correspond to the given level, or to the levels
	// immediately above or below (level+1) and (level-1).
	*/
	 __device__  void relevel_stash(unsigned int level){

		if( ! StackType::FLAT ){

			// TODO: Implement for non-flat stack

		}

	}





	/*
	// Dumps all full links not corresponding to the current execution level. Furthermore, should the
	// remaining links still put the stash over the given threshold occupancy, links will be further
	// removed in the order: full links at the current level, partial links not at the current level,
	// partial links at the current level. 
	*/
	 __device__  void spill_stash(unsigned int threashold){

		unsigned int active =__activemask();
		__syncwarp(active);


		
	#if DEF_STACK_MODE == 0

		
		
		if(util::current_leader() && (_grp_ctx.stash_count > threashold)){


			unsigned int spill_count = _grp_ctx.stash_count - threashold;
			int delta = 0;
			fill_stash_links(spill_count);
			
			QueueType queue;
			queue.pair.data = QueueType::null;
			unsigned int partial_iter = 0;
			bool has_full_slots = true;
			//printf("{Spilling to %d}",threashold);
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
						if(_grp_ctx.partial_map[partial_iter] != STASH_SIZE){
							slot = _grp_ctx.partial_map[partial_iter];
							//printf("{Spilling partial}");
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
				if(_grp_ctx.stash_count <= threashold){
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
			unsigned int dump_count = (_grp_ctx.stash_count > threshold) ? _grp_ctx.stash_count - threshold : 0;
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
			for(unsigned int i=0; i < _grp_ctx.stash_count; i++){
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



	 __device__  void async_call_stash_dump(OpDisc func_id, int depth_delta, unsigned int delta){

		/*
		// Make room to queue incoming promises, if there isn't enough room already.
		*/
		#if 0
		if(_grp_ctx.link_stash_count < 2){
			fill_stash_links(2);
		}

		if(_grp_ctx.stash_count >= (STASH_SIZE-2)){
			spill_stash(STASH_SIZE-3);
		}
		#else
		unsigned int depth = (unsigned int) (_grp_ctx.level + depth_delta);
		unsigned int left_jump = partial_map_index(func_id,depth,_grp_ctx.level);
		/*
		unsigned int space = 0;
		if( left_jump != PART_ENTRY_COUNT ){
			unsigned int left_idx = _grp_ctx.partial_map[left_jump];	
			if( left_idx != STASH_SIZE ){
				space = WORK_GROUP_SIZE - _grp_ctx.stash[left_idx].count;
			}
		}
		*/
		if( (_grp_ctx.stash_count >= (STASH_SIZE-2)) ) { //&& (space < delta) ){
			if(_grp_ctx.link_stash_count < 2){
				fill_stash_links(2);
			}
			//printf("{Spilling for call.}");
			spill_stash(STASH_SIZE-3);
		}
		#endif

	}


	 __device__  void async_call_stash_prep(OpDisc func_id, int depth_delta, unsigned int delta,
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
				//db_printf("A\n");
				left = _grp_ctx.partial_map[left_jump];
			}

			unsigned int left_count;
			if(left == STASH_SIZE){
				//db_printf("B\n");
				left = claim_empty_slot();
				_grp_ctx.stash_count += 1;
				db_printf("Updated stash count: %d\n",_grp_ctx.stash_count);
				_grp_ctx.stash[left].id    = func_id;
				_grp_ctx.partial_map[left_jump] = left;
				left_count = 0;
			} else {
				left_count = _grp_ctx.stash[left].count;
			}

			if ( (left_count + delta) > WORK_GROUP_SIZE ){
				//db_printf("C\n");
				right = claim_empty_slot();
				_grp_ctx.stash_count += 1;
				db_printf("Updated stash count: %d\n",_grp_ctx.stash_count);
				_grp_ctx.stash[right].count = left_count+delta - WORK_GROUP_SIZE;
				_grp_ctx.stash[right].id    = func_id;
				insert_full_slot(left);
				_grp_ctx.partial_map[left_jump] = right;
				_grp_ctx.stash[left].count = WORK_GROUP_SIZE;
			} else if ( (left_count + delta) == WORK_GROUP_SIZE ){
				//db_printf("D\n");
				_grp_ctx.partial_map[left_jump] = STASH_SIZE;
				insert_full_slot(left);
				_grp_ctx.stash[left].count = WORK_GROUP_SIZE;
			} else {
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

		async_call_stash_dump(func_id, depth_delta, delta);

		__shared__ unsigned int left, left_start, right;


		async_call_stash_prep(func_id,depth_delta,delta,left,left_start,right);
	

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
	template<typename TYPE>
	 __device__  void async_call_cast(int depth_delta, Promise<TYPE> param_value){

		beg_time(7);
		unsigned int active = __activemask();

		/*
		// Calculate how many promises are being queued as well as the assigned index of the 
		// current thread's promise in the write to the stash.
		*/
		unsigned int index = util::warp_inc_scan();
		unsigned int delta = util::active_count();

		beg_time(8);
		async_call_stash_dump(Lookup<TYPE>::type::DISC, depth_delta, delta);
		end_time(8);

		__shared__ unsigned int left, left_start, right;


		beg_time(9);
		async_call_stash_prep(Lookup<TYPE>::type::DISC,depth_delta,delta,left,left_start,right);
		end_time(9);
		
		/*
		// Write the promise into the appropriate part of the stash, writing into the left link 
		// when possible and spilling over into the right link when necessary.
		*/
		__syncwarp(active);
		if( (left_start + index) >= WORK_GROUP_SIZE ){
			//db_printf("Overflow: id: %d, left: %d, left_start: %d, index: %d\n",threadIdx.x,left,left_start,index);
			_grp_ctx.stash[right].promises[left_start+index-WORK_GROUP_SIZE].template cast<TYPE>() = param_value;
		} else {
			//db_printf("Non-overflow: id: %d, left: %d, left_start: %d, index: %d\n",threadIdx.x,left,left_start,index);
			_grp_ctx.stash[left].promises[left_start+index].template cast<TYPE>() = param_value;
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

		
		db_printf("\n\nprior stash count: %d\n\n\n",_grp_ctx.stash_count);
		//*
		for(unsigned int i=0; i< add_count; i++){
			//PromiseUnionType promise = _dev_ctx.arena[the_index].promises[i];
			//async_call(func_id,0,promise);
			async_call(func_id,0, _dev_ctx.arena[the_index].promises[i] );
		}
		// */
		//PromiseType promise = _dev_ctx.arena[the_index].data.data[0];
		//async_call(func_id,0,promise);

		db_printf("\n\nafter stash count: %d\n\n\n",_grp_ctx.stash_count);


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
			
			unsigned int gather_count = (threashold < _grp_ctx.stash_count) ? 0  : threashold - _grp_ctx.stash_count;
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
				if( get_frame(_grp_ctx.level).children_residents != 0 ){
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
					pull_any = (right_half(current_frame.children_residents) == 0);
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
				     && (_grp_ctx.stash_count < threashold)
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
				if( _grp_ctx.stash_count >= threashold ){
					break;
				}
				#endif

			}
		




			#ifdef PARACON
			if(_grp_ctx.busy && (_grp_ctx.stash_count == 0) && (taken == 0) ){
			#else
			if(_grp_ctx.busy && (_grp_ctx.stash_count == 0)){
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

			if(_grp_ctx.full_head != STASH_SIZE){
				_grp_ctx.exec_head = claim_full_slot();
				_grp_ctx.stash_count -= 1;
				result = true;
				//db_printf("Found full slot.\n");
			} else {
				//db_printf("Looking for partial slot...\n");
				unsigned int best_id   = PART_ENTRY_COUNT;
				unsigned int best_slot = STASH_SIZE;
				unsigned int best_count = 0;
				for(int i=0; i < FN_ID_COUNT; i++){
					unsigned int slot = _grp_ctx.partial_map[i];
					
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
					_grp_ctx.partial_map[best_id] = STASH_SIZE;
					_grp_ctx.stash_count -=1;
				}
			}

		}

		__syncwarp(active);
		return result;

	}




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


		beg_time(1);
		
		if ( ( ((_dev_ctx.stack->frames[0].children_residents) & 0xFFFF ) > (gridDim.x*blockIdx.x*2) ) && (_grp_ctx.full_head == STASH_SIZE) ) { 
			fill_stash(STASH_SIZE-2,false);
		}
		
		end_time(1);

		beg_time(5);
		if ( _grp_ctx.can_make_work && (_grp_ctx.full_head == STASH_SIZE) ) {
			_grp_ctx.can_make_work = PROGRAM_SPEC::make_work(*this);
			if( util::current_leader() && (! _grp_ctx.busy ) && ( _grp_ctx.stash_count != 0 ) ){
				unsigned int depth_live = atomicAdd(&(_dev_ctx.stack->depth_live),1);
				_grp_ctx.busy = true;
				//printf("{made self busy %d depth_live=(%d,%d)}",blockIdx.x,(depth_live & 0xFFFF0000)>>16u, depth_live & 0xFFFF);
			}
		}
		end_time(5);


		#if 1

		/*
		if ( _grp_ctx.full_head == STASH_SIZE ) {
			fill_stash(STASH_SIZE-2);
		}
		*/
		#else
		if(_grp_ctx.full_head == STASH_SIZE){
			if( !_grp_ctx.scarce_work ){
				fill_stash(STASH_SIZE-2);
				if( util::current_leader() && (_grp_ctx.full_head == STASH_SIZE) ){
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

		beg_time(2);
		if( !advance_stash_iter() ){
			/*
			// No more work exists in the stash, so try to fetch it from the stack.
			*/
			beg_time(10);
			fill_stash(STASH_SIZE-2,true);
			end_time(10);

			if( _grp_ctx.keep_running && !advance_stash_iter() ){
				/*
				// REALLY BAD: The fill_stash function successfully, however 
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
	

		if(threadIdx.x == 0){

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
	__device__ void init(){

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
				_dev_ctx.stack->frames[target_level].children_residents = 0u;
			} else {
				_dev_ctx.stack->frames[target_level].pool.queues[frame_index].pair.data = QueueType::null;
			}

		}


		#ifdef LAZY_LINK

		/*
		// Initialize the pool, assigning empty queues to each queue slot.
		*/
		for(unsigned int index = _thd_ctx.thread_id; index < POOL_SIZE; index+=worker_count ){	
			
			_dev_ctx.pool->queues[index].pair.data = QueueType::null;

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

		if(util::current_leader()){
			//printf("\n\n\nInitial frame zero resident count is: %d\n\n\n",_dev_ctx.stack->frames[0].children_residents);
		}	

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
				unsigned int children_residents = host_stack->frames[i].children_residents;
				unsigned int children  = (children_residents >> 16) & 0xFFFF;
				unsigned int residents = children_residents & 0xFFFF;
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
	 __device__  void async(ARGS... args){
		async_call_cast<TYPE>(0,Promise<TYPE>(args...));
	}

	template<typename TYPE,typename... ARGS>
	__device__  void sync(ARGS... args){
		immediate_call_cast<TYPE>(Promise<TYPE>(args...));
	}


	template<typename TYPE>
	__device__ float queue_fill_fraction()
	{
		return NAN;

	}


};



/*
// These functions are here just to trampoline into the actual main functions for a given program.
// This is done because structs/classes may not have global member functions.
*/
template<typename ProgType>
__global__ void _dev_init(typename ProgType::DeviceContext _dev_ctx, typename ProgType::DeviceState device) {
	
		
	__shared__ typename ProgType::GroupContext _grp_ctx;
	__shared__ typename ProgType::GroupState   group;


	typename ProgType::ThreadContext _thd_ctx;
	typename ProgType::ThreadState   thread;

	ProgType prog(_dev_ctx,_grp_ctx,_thd_ctx,device,group,thread);

	prog.init();
}


template<typename ProgType>
__global__ void _dev_exec(typename ProgType::DeviceContext _dev_ctx, typename ProgType::DeviceState device, size_t cycle_count) {
	
	__shared__ typename ProgType::GroupContext _grp_ctx;
	__shared__ typename ProgType::GroupState   group;

	typename ProgType::ThreadContext _thd_ctx;
	typename ProgType::ThreadState   thread;

	ProgType prog(_dev_ctx,_grp_ctx,_thd_ctx,device,group,thread);

	prog.exec(cycle_count);
}

























template<typename PROGRAM_SPEC>
class EventProgram
{


	public:


	typedef EventProgram<PROGRAM_SPEC> ProgramType;

	#define MEMBER_GUARD(NAME,DEFAULT) \
		struct NAME##Default { typedef unsigned int NAME; }; \
		template<class TYPE> using  NAME##Check = typename TYPE::NAME; \
		typedef typename type_switch<NAME##Default,NAME##Check,PROGRAM_SPEC>::NAME NAME;

	/*
	struct AdrTypeDefault { typedef unsigned int AdrType; };
	template<class TYPE> using  AdrTypeCheck = typename TYPE::AdrType;
	typedef typename type_or<AdrTypeDefault,AdrTypeCheck,PROGRAM_SPEC>::AdrType AdrType;
	*/

	MEMBER_GUARD(    AdrType,unsigned int)
	MEMBER_GUARD(      OpSet,   OpUnion<>)
	MEMBER_GUARD(DeviceState,   VoidState)
	MEMBER_GUARD( GroupState,   VoidState)
	MEMBER_GUARD(ThreadState,   VoidState)

	#undef MEMBER_GUARD

	typedef PromiseUnion<OpSet> PromiseUnionType;

	template<typename TYPE>
	struct Lookup { typedef typename PromiseUnionType::Lookup<TYPE>::type type; };


	#define SIZE_T_GUARD(NAME,DEFAULT) \
		struct NAME##Default { static const size_t NAME = 16; }; \
		template<class TYPE> using  NAME##Check = decltype( TYPE::NAME ); \
		static const size_t NAME = type_switch<NAME##Default,NAME##Check,PROGRAM_SPEC>::NAME;
	
	SIZE_T_GUARD(GROUP_SIZE,32)

	#undef SIZE_T_GUARD



	static const size_t       WORK_GROUP_SIZE  = GROUP_SIZE;

	/*
	// A set of halting condition flags
	*/
	static const unsigned int BAD_FUNC_ID_FLAG	= 0x00000001;
	static const unsigned int COMPLETION_FLAG	= 0x80000000;


	/*
	// The number of async functions present in the program.
	*/
	static const unsigned char FN_ID_COUNT = PromiseUnionType::COUNT;

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

		unsigned int*                               checkout;
		util::iter::IOBuffer<PromiseUnionType,AdrType>*   event_io[PromiseUnionType::COUNT];
	};


	/*
	// Instances wrap around their program scope's DeviceContext. These differ from a program's
	// DeviceContext object in that they perform automatic deallocation as soon as they drop
	// out of scope.
	*/
	struct Instance {

		
		util::host::DevBuf<unsigned int> checkout;
		util::host::DevObj<util::iter::IOBuffer<PromiseUnionType>> event_io[PromiseUnionType::COUNT];
		DeviceState device_state;		

		__host__ Instance (size_t io_size, DeviceState gs)
			: device_state(gs)
		{
			for( unsigned int i=0; i<PromiseUnionType::COUNT; i++){
				event_io[i] = util::host::DevObj<util::iter::IOBuffer<PromiseUnionType>>(io_size);
			}
			checkout<< 0u;
		}

		__host__ DeviceContext to_context(){

			DeviceContext result;
			
			result.checkout = checkout;
			for( unsigned int i=0; i<PromiseUnionType::COUNT; i++){
				result.event_io[i] = event_io[i];
			}

			return result;

		}

		__host__ bool complete(){

			for( unsigned int i=0; i<PromiseUnionType::COUNT; i++){
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

	DeviceContext & _dev_ctx;
	GroupContext  & _grp_ctx;
	ThreadContext & _thd_ctx;

	public:

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
				for(unsigned int i=0; i < PromiseUnionType::COUNT; i++){
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
				while(PROGRAM_SPEC::make_work(*this)){}
				break;
			}

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


		__syncthreads();

		PROGRAM_SPEC::finalize(*this);
		
		__threadfence();
		__syncthreads();

		if( threadIdx.x == 0 ){
			unsigned int checkout_index = atomicAdd(_dev_ctx.checkout,1);
			//printf("{%d}",checkout_index);
			if( checkout_index == (gridDim.x - 1) ){
				//printf("{Final}");
				atomicExch(_dev_ctx.checkout,0);
				 for(unsigned int i=0; i < PromiseUnionType::COUNT; i++){
					 _dev_ctx.event_io[i]->flip();
				 }
				 
			}
		}



	}


	template<typename TYPE,typename... ARGS>
	__device__  void async(ARGS... args){
		async_call_cast<TYPE>(0,Promise<TYPE>(args...));
	}

	template<typename TYPE,typename... ARGS>
	__device__  void sync(ARGS... args){
		immediate_call_cast<TYPE>(Promise<TYPE>(args...));
	}
	
	template<typename TYPE>
	__device__ float queue_fill_fraction()
	{
		return _dev_ctx.event_io[Lookup<TYPE>::type::DISC]->output_fill_fraction_sync();

	}
};








/*
// These functions unwrap an instance into its device context and passes it to the responsible
// kernel.
*/
template<typename ProgType>
__host__ void init(typename ProgType::Instance& instance,size_t group_count) {
	_dev_init<ProgType><<<group_count,ProgType::WORK_GROUP_SIZE>>>(instance.to_context(),instance.device_state);
}
template<typename ProgType>
__host__ void exec(typename ProgType::Instance& instance,size_t group_count, size_t cycle_count) {
	_dev_exec<ProgType><<<group_count,ProgType::WORK_GROUP_SIZE>>>(instance.to_context(),instance.device_state,cycle_count);
}


