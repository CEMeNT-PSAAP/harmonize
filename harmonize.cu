



//#define DEBUG_PRINT
//#define RACE_COND_PRINT
//#define QUEUE_PRINT

#define INF_LOOP_SAFE


#ifdef QUEUE_PRINT
	#define q_printf  printf
#else
	#define q_printf(fmt, ...) ;
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



#include "util.cu"


enum class Fn;


template <Fn ID>
struct PromiseType;


template <typename GLOBAL_STATE, typename GROUP_STATE, typename THREAD_STATE>
struct ProgramStateDef
{
	typedef GLOBAL_STATE GlobalState;
	typedef GROUP_STATE  GroupState;
	typedef THREAD_STATE ThreadState;
};


template <typename PROG_TYPE, Fn ID>
 __device__ void  promise_eval (
	typename PROG_TYPE::GlobalContext &,
	typename PROG_TYPE::GroupContext  &,
	typename PROG_TYPE::ThreadContext &,
	typename PROG_TYPE::GlobalState   &,
	typename PROG_TYPE::GroupState    &,
	typename PROG_TYPE::ThreadState   &,
	typename PromiseType<ID>::ParamType
);




template <Fn... IDS>
struct PromiseCount;

template<>
struct PromiseCount<>
{
	static const unsigned char value = 0;
};


template<Fn HEAD, Fn... TAIL>
struct PromiseCount<HEAD, TAIL...>
{
	static const unsigned char value = PromiseCount<TAIL...>::value + 1;
};


template <Fn... IDS>
union PromiseUnion;


template <>
union PromiseUnion <> {

	template <typename PROG_TYPE, Fn ID>
	__host__  __device__ void rigid_eval(
		typename PROG_TYPE::GlobalContext & global_context,
		typename PROG_TYPE::GroupContext  & group_context,
		typename PROG_TYPE::ThreadContext & thread_context,
		typename PROG_TYPE::GlobalState   & global_state,
		typename PROG_TYPE::GroupState    & group_state,
		typename PROG_TYPE::ThreadState   & thread_state
	) {
		return;
	}

	template <typename PROG_TYPE>
	__host__  __device__ void loose_eval(
		Fn Id,
		typename PROG_TYPE::GlobalContext & global_context,
		typename PROG_TYPE::GroupContext  & group_context,
		typename PROG_TYPE::ThreadContext & thread_context,
		typename PROG_TYPE::GlobalState   & global_state,
		typename PROG_TYPE::GroupState    & group_state,
		typename PROG_TYPE::ThreadState   & thread_state
	) {
		return;
	}

};

template <Fn HEAD, Fn... TAIL>
union PromiseUnion<HEAD, TAIL...>
{

	typename PromiseType<HEAD>::ParamType head_form;
	PromiseUnion<TAIL...> tail_form;

	public:

	typedef PromiseCount<HEAD,TAIL...> Count;

	template <Fn ID>
	__host__  __device__ typename std::enable_if< (ID==HEAD), typename PromiseType<ID>::ParamType& >::type
	cast() {	
		return head_form;
	}

	template <Fn ID>
	__host__  __device__ typename std::enable_if< (ID>HEAD), typename PromiseType<ID>::ParamType& >::type
	cast(){	
		return tail_form.cast<ID>();
	}

	template <Fn ID>
	__host__  __device__ typename std::enable_if< (ID<HEAD), void >::type
	cast (){	
		static_assert( ID<HEAD, "Function ID does not exist in union" );
	}


	template <typename PROG_TYPE, Fn ID>
	__host__  __device__ typename std::enable_if< (ID==HEAD), void >::type
	rigid_eval(
		typename PROG_TYPE::GlobalContext & global_context,
		typename PROG_TYPE::GroupContext  & group_context,
		typename PROG_TYPE::ThreadContext & thread_context,
		typename PROG_TYPE::GlobalState   & global_state,
		typename PROG_TYPE::GroupState    & group_state,
		typename PROG_TYPE::ThreadState   & thread_state
	) {
		promise_eval<PROG_TYPE,ID>(
			global_context,
			group_context,
			thread_context,
			global_state,
			group_state,
			thread_state,
			head_form
		);
	}

	template <typename PROG_TYPE, Fn ID>
	__host__  __device__ typename std::enable_if< (ID>HEAD), void >::type
	rigid_eval(
		typename PROG_TYPE::GlobalContext & global_context,
		typename PROG_TYPE::GroupContext  & group_context,
		typename PROG_TYPE::ThreadContext & thread_context,
		typename PROG_TYPE::GlobalState   & global_state,
		typename PROG_TYPE::GroupState    & group_state,
		typename PROG_TYPE::ThreadState   & thread_state
	) {
		tail_form.rigid_eval<PROG_TYPE,ID>(
			global_context,
			group_context,
			thread_context,
			global_state,
			group_state,
			thread_state
		);
	}

	template <typename PROG_TYPE, Fn ID>
	__host__  __device__ typename std::enable_if< (ID<HEAD), void >::type
	rigid_eval() {
		static_assert( ID<HEAD, "Function ID does not exist in union" );
	}


	template <typename PROG_TYPE>
	__host__  __device__ void loose_eval (
		Fn id,
		typename PROG_TYPE::GlobalContext & global_context,
		typename PROG_TYPE::GroupContext  & group_context,
		typename PROG_TYPE::ThreadContext & thread_context,
		typename PROG_TYPE::GlobalState   & global_state,
		typename PROG_TYPE::GroupState    & group_state,
		typename PROG_TYPE::ThreadState   & thread_state
	) {
		if(id == HEAD){
			promise_eval<PROG_TYPE,HEAD>(
				global_context,
				group_context,
				thread_context,
				global_state,
				group_state,
				thread_state,
				head_form
			);
		} else {
			tail_form. template loose_eval<PROG_TYPE>(
				id,
				global_context,
				group_context,
				thread_context,
				global_state,
				group_state,
				thread_state
			);
		}

	}

};




template <typename ADR_INNER_TYPE>
struct WorkLinkAdr
{

	typedef ADR_INNER_TYPE AdrType;

	static const AdrType null = std::numeric_limits<AdrType>::max();

	AdrType adr;


	WorkLinkAdr<AdrType>() = default;

	 __device__ WorkLinkAdr<AdrType>(AdrType adr_val) : adr(adr_val) {}

	 __device__ bool operator==(const WorkLinkAdr<AdrType>& link_adr){
		return (adr == link_adr.adr);
	}


	 __device__ bool is_null(){
		return (adr == null);
	}


};



template <typename ADR_TYPE>
struct WorkQueue;


template <typename ADR_TYPE>
struct WorkQueue <WorkLinkAdr<ADR_TYPE>>
{

	typedef ADR_TYPE AdrType;
	typedef WorkLinkAdr<AdrType> LinkAdrType;
	typedef WorkQueue<LinkAdrType> Self;

	typedef typename util::PairEquivalent<AdrType>::Type QueueType;
	util::PairPack<AdrType> pair;

	static const QueueType null = std::numeric_limits<QueueType>::max();

	 __device__ LinkAdrType get_head(){
		return pair.get_left();
	}

	/*
	// This function extracts the tail of the given queue
	*/
	 __device__ LinkAdrType get_tail(){
		return pair.get_right();
	}

	/*
	// Enable default constructor
	*/
	WorkQueue<WorkLinkAdr<ADR_TYPE>>() = default;

	/*
	// This function concatenates two link addresses into a queue
	*/
	 __device__ WorkQueue<WorkLinkAdr<ADR_TYPE>>
	(LinkAdrType head, LinkAdrType tail)
	{
		pair = util::PairPack<AdrType>(head.adr,tail.adr);
	}

	/*
	// This function sets the head of the queue to the given link address
	*/
	 __device__ void set_head(LinkAdrType head){
		pair.set_left(head.adr);
	}

	/*
	// This function sets the tail of the queue to the given link address
	*/
	 __device__ void set_tail(LinkAdrType tail){
		pair.set_right(tail.adr);
	}

	/*
	// Returns true if and only if queue is empty
	*/
	 __device__ bool is_null(){
		return (pair.data == Self::null);
	}


};




template <typename PROMISE_UNION, typename ADR_TYPE, size_t GROUP_SIZE>
struct WorkLink
{

	typedef ADR_TYPE AdrType;
	typedef PROMISE_UNION PromiseType;

	PromiseType    promises[GROUP_SIZE];

	AdrType        next;
	unsigned int   meta_data;
	Fn             id;
	unsigned short count;


	/*
	// Zeros out a link, giving it a thunk count of zero, a null function ID, a null pointer for
	// the next link and a depth of zero.
	*/
	 __device__ void empty(AdrType next_adr){

		next	= next_adr;
		id	= static_cast<Fn>(PromiseType::Count::value);
		count	= 0;

	}


};






template <typename ADR_TYPE, typename LINK_TYPE>
struct WorkArena
{

	typedef LINK_TYPE LinkType;
	typedef ADR_TYPE  LinkAdrType;

	static const size_t max_size = ADR_TYPE::null;
	size_t     size;
	LinkType *links;

	 __device__ LinkType& operator[](LinkAdrType adr){
		return links[adr.adr];
	}

};




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

	unsigned int	event_com;
	unsigned int	depth_live;
	FRAME_TYPE frames[STACK_SIZE];

};

template<typename FRAME_TYPE>
struct WorkStack<FRAME_TYPE, 0>
{
	static const bool   FLAT       = true;
	
	static const size_t PART_MULT  = 1;
	static const size_t NULL_LEVEL = 1;
	
	unsigned int	event_com;
	unsigned int	depth_live;
	FRAME_TYPE frames[1];

};




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


struct VoidState {};


template<
	typename PROMISE_UNION,
	typename PROGRAM_STATE,
	typename ADR_TYPE = unsigned int,
	size_t   STASH_SIZE = 16,
	size_t   FRAME_SIZE = 32,
	size_t   POOL_SIZE  = 32,
	size_t   GROUP_SIZE = 32,
	size_t   STACK_SIZE = 0
>
struct HarmonizeProgram;

template<
	Fn... FN_IDS,
	typename PROGRAM_STATE,
	typename ADR_TYPE,
	size_t   STASH_SIZE,
	size_t   FRAME_SIZE,
	size_t   POOL_SIZE,
	size_t   GROUP_SIZE,
	size_t   STACK_SIZE
>
struct HarmonizeProgram<
	PromiseUnion<FN_IDS...>,
	PROGRAM_STATE,
	ADR_TYPE,
	STASH_SIZE,
	FRAME_SIZE,
	POOL_SIZE,
	GROUP_SIZE,
	STACK_SIZE
>
{



	typedef struct HarmonizeProgram<
			PromiseUnion<FN_IDS...>,
			PROGRAM_STATE,
			ADR_TYPE,
			STASH_SIZE,
			FRAME_SIZE,
			POOL_SIZE,
			GROUP_SIZE,
			STACK_SIZE
		> ProgramType;


	/*
	// During system verification/debugging, this will be used as a cutoff to prevent infinite
	// looping
	*/
	static const unsigned int RETRY_LIMIT      = 32;
	

	static const size_t       WORK_GROUP_SIZE  = GROUP_SIZE;

	/*
	// A set of halting condition flags
	*/
	static const unsigned int BAD_FUNC_ID_FLAG	= 0x00000001;
	static const unsigned int STASH_FAIL_FLAG	= 0x00000002;
	static const unsigned int COMPLETION_FLAG	= 0x80000000;


	/*
	// The types representing the global, per-group, and per-thread information that needs to
	// be tracked for the developer's program.
	*/
	typedef typename PROGRAM_STATE::GlobalState   GlobalState;
	typedef typename PROGRAM_STATE::GroupState    GroupState;
	typedef typename PROGRAM_STATE::ThreadState   ThreadState;


	typedef PromiseUnion <FN_IDS...>  PromiseUnionType;
	typedef WorkLinkAdr  <ADR_TYPE>                 LinkAdrType;
	typedef WorkQueue    <LinkAdrType>              QueueType;
	typedef WorkFrame    <QueueType,FRAME_SIZE>     FrameType;
	typedef WorkStack    <FrameType,STACK_SIZE>     StackType;
	typedef WorkPool     <QueueType,POOL_SIZE>      PoolType;

	typedef WorkLink     <PromiseUnionType, LinkAdrType, WORK_GROUP_SIZE> LinkType;
	
	typedef WorkArena    <LinkAdrType,LinkType>     ArenaType;


	/*
	// The number of async functions present in the program.
	*/
	static const unsigned char FN_ID_COUNT = PromiseCount<FN_IDS...>::value;


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

		ThreadState	thread_state;

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

		unsigned char			link_stash_count; // Number of global-space links stored

		LinkType			stash[STASH_SIZE];
		LinkAdrType			link_stash[STASH_SIZE];
		
		
		int				SM_promise_delta;
		unsigned long long int		work_iterator;


		GroupState			group_state;

	};


	/*
	// This struct represents the entire set of data structures that must be stored in global
	// memory to track the state of the program defined by the developer as well as the state
	// of the context which is driving execution.
	*/
	struct GlobalContext {

		typedef		ProgramType	ParentProgramType;

		ArenaType	arena;
		PoolType*	pool;
		StackType*	stack;

		GlobalState	global_state;

	};


	/*
	// Instances wrap around their program scope's GlobalContext. These differ from a program's
	// GlobalContext object in that they perform automatic deallocation as soon as they drop
	// out of scope.
	*/
	struct Instance {

		size_t arena_size;
		util::DevBuf<LinkType>  arena;
		util::DevBuf<PoolType>  pool;
		util::DevBuf<StackType> stack;
		GlobalState global_state;		

		__host__ Instance (size_t arsize, GlobalState gs)
			: arena(arsize)
			, pool (1)
			, stack(1)
			, arena_size(arsize)
			, global_state(gs)
		{ }

		__host__ GlobalContext to_context(){

			GlobalContext result;
			result.arena.size   = arena_size;
			result.arena.links  = arena;
			result.pool         = pool ;
			result.stack        = stack;
			result.global_state = global_state;

			return result;

		}

	};



	#define _CTX_ARGS GlobalContext &glb, GroupContext &grp, ThreadContext &thd
	#define _STATE_ARGS GlobalState &global, GroupState &group, ThreadState &thread
	#define _CTX_REFS glb,grp,thd
	#define _STATE_REFS glb.global_state,grp.group_state,thd.thread_state


	/*
	// To be defined by developer. These can be given an empty definition, if desired.
	*/
	 __device__ static void        initialize(_CTX_ARGS, _STATE_ARGS);
	 __device__ static void        finalize  (_CTX_ARGS, _STATE_ARGS);
	 __device__ static bool        make_work (_CTX_ARGS, _STATE_ARGS);

	/*
	// Returns an index into the partial map of a group based off of a function id and a depth. If
	// an invalid depth or function id is used, PART_ENTRY_COUNT is returned.
	*/
	 __device__ static unsigned int partial_map_index(
		Fn func_id,
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
	 __device__ static void init_group(GroupContext& grp){

		unsigned int active = __activemask();

		__syncwarp(active);

		if(util::current_leader()){

			if( StackType::FLAT ){
				grp.level = 0;
			} else {
				grp.level = StackType::NULL_LEVEL;
			}

			grp.stash_count = 0;
			grp.link_stash_count = 0;
			grp.keep_running = true;
			grp.busy 	 = false;
			grp.can_make_work= true;
			grp.exec_head    = STASH_SIZE;
			grp.full_head    = STASH_SIZE;
			grp.empty_head   = 0;
			grp.work_iterator= 0;
			grp.scarce_work  = false;

			for(unsigned int i=0; i<STASH_SIZE; i++){
				grp.stash[i].empty(i+1);
			}
				
			for(unsigned int i=0; i<PART_ENTRY_COUNT; i++){
				grp.partial_map[i] = STASH_SIZE;
			}

			grp.SM_promise_delta = 0;
			
		}

		__syncwarp(active);

	}

	/*
	// Initializes the local state of a thread, which is just the global id of the thread and the
	// state used by the thread to generate random numbers for stochastic choices needed to manage
	// the runtime state.
	*/
	 __device__ static ThreadContext init_thread(){

		ThreadContext result;

		result.thread_id  = (blockIdx.x * blockDim.x) + threadIdx.x;
		result.rand_state = result.thread_id;

		return result;
	}


	/*
	// Sets the bits in the event_com field of the stack according to the given flag bits.
	*/
	 __device__ static void set_flags(_CTX_ARGS, unsigned int flag_bits){

		atomicOr(&glb.stack->event_com,flag_bits);

	}


	/*
	// Returns the current highest level in the stack. Given that this program is highly parallel,
	// this number inherently cannot be trusted. By the time the value is fetched, the stack could
	// have a different height or the thread that set the height may not have deposited links in the
	// corresponding level yet.
	*/
	 __device__ static unsigned int highest_level(_CTX_ARGS){

		return left_half(glb.stack->depth_live);

	}


	/*
	// Returns a reference to the frame at the requested level in the stack.
	*/
	 __device__ static FrameType& get_frame(_CTX_ARGS, unsigned int level){

		return glb.stack->frames[level];

	}


	/*
	// Joins two queues such that the right queue is now at the end of the left queue.
	//
	// WARNING: NOT THREAD SAFE. Only use this on queues that have been claimed from the stack
	// atomically. If not, one of the queues manipulated with this function will almost certainly
	// become malformed at some point. Woe betide those that do not heed this dire message.
	*/
	 __device__ static QueueType join_queues(_CTX_ARGS, QueueType left_queue, QueueType right_queue){

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
			LinkType& left_tail = glb.arena[left_tail_adr];

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
	 __device__ static LinkAdrType pop_front(_CTX_ARGS, QueueType queue){

		LinkAdrType result;
		/*
		// Don't try unless the queue is non-null
		*/
		if( queue.is_null() ){
			result.adr = LinkAdrType::null;
		} else {
			result = queue.get_head();
			LinkAdrType next = glb.arena[result].next;
			queue.set_head(next);
			if(next.adr == LinkAdrType::null){
				queue.set_tail(next);
			} else if ( queue.get_tail() == result ){
				q_printf("\n\nFinal link does not have a null next.\n\n");
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
	 __device__ static void push_back(_CTX_ARGS, QueueType& queue, LinkAdrType link_adr){

		glb.arena[link_adr].next = LinkAdrType::null;
		if( queue.is_null() ){
			queue = QueueType(link_adr,link_adr);
		} else {
			LinkAdrType tail = queue.get_tail();
			glb.arena[tail].next = link_adr;
			queue.set_tail(link_adr);
		}

	}



	/*
	// Attempts to pull a queue from a range of queue slots, trying each slot starting from the given
	// starting index onto the end of the range and then looping back from the beginning. If, after
	// trying every slot in the range, no non-null queue was obtained, a QueueType::null value is returned.
	*/
	 __device__ static QueueType pull_queue(QueueType* src, unsigned int start_index, unsigned int range_size, unsigned int& src_index){

		QueueType result;
		
		
		/*
		// First iterate from the starting index to the end of the queue range, attempting to
		// claim a non-null queue until either there are no more slots to try, or the atomic
		// swap successfuly retrieves something.
		*/
		for(unsigned int i=start_index; i < range_size; i++){
			result.pair.data = atomicExch(&(src[i].pair.data),QueueType::null);
			if( ! result.is_null() ){
				src_index = i;
				return result;
			}
		}

		/*
		// Continue searching from the beginning of the range to just before the beginning of the
		// previous scan.
		*/
		for(unsigned int i=0; i < start_index; i++){
			result.pair.data = atomicExch(&(src[i].pair.data),QueueType::null);
			if( ! result.is_null() ){
				src_index = i;
				return result;
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
	 __device__ static void push_queue(_CTX_ARGS, QueueType& dest, QueueType queue){

		#ifdef INF_LOOP_SAFE
		while(true)
		#else
		for(int i=0; i<RETRY_LIMIT; i++)
		#endif
		{
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
				q_printf("Ugh. We got queue (%d,%d) when trying to push a queue\n",get_head(swap),get_tail(swap));
				QueueType other_swap;
				other_swap.pair.data = atomicExch(&dest.pair.data,QueueType::null); 
				queue = join_queues(_CTX_REFS,other_swap,swap);
				q_printf("Merged it to form queue (%d,%d)\n",get_head(queue),get_tail(queue));
			} else {
				q_printf("Finally pushed (%d,%d)\n",get_head(queue),get_tail(queue));
				break;
			}
		}


	}




	/*
	// Claims a link from the link stash. If no link exists in the stash, LinkAdrType::null is returned.
	*/
	 __device__ static LinkAdrType claim_stash_link(_CTX_ARGS){

		LinkAdrType link = LinkAdrType::null;
		unsigned int count = grp.link_stash_count;
		if(count > 0){
			link = grp.link_stash[count-1];
			grp.link_stash_count = count - 1;
		}
		q_printf("New link stash count: %d\n",grp.link_stash_count);
		return link;

	}



	/*
	// Inserts an empty slot into the stash. This should only be called if there is enough space in
	// the link stash.
	*/
	 __device__ static void insert_stash_link(_CTX_ARGS, LinkAdrType link){

		unsigned int count = grp.link_stash_count;
		grp.link_stash[count] = link;
		grp.link_stash_count = count + 1;
		q_printf("New link stash count: %d\n",grp.link_stash_count);

	}




	/*
	// Claims an empty slot from the stash and returns its index. If no empty slot exists in the stash,
	// then STASH_SIZE is returned.
	*/
	 __device__ static unsigned int claim_empty_slot(_CTX_ARGS){

		unsigned int slot = grp.empty_head;
		if(slot != STASH_SIZE){
			grp.empty_head = grp.stash[slot].next.adr;
			db_printf("EMPTY: %d << %d\n",slot,grp.empty_head);
		}
		return slot;

	}


	/*
	// Inserts an empty slot into the stash. This should only be called if there is enough space in
	// the link stash.
	*/
	 __device__ static void insert_empty_slot(_CTX_ARGS, unsigned int slot){

		grp.stash[slot].next.adr = grp.empty_head;
		db_printf("EMPTY: >> %d -> %d\n",slot,grp.empty_head);
		grp.empty_head = slot;

	}


	/*
	// Claims a full slot from the stash and returns its index. If no empty slot exists in the stash,
	// then STASH_SIZE is returned.
	*/
	 __device__ static unsigned int claim_full_slot(_CTX_ARGS){

		unsigned int slot = grp.full_head;
		if(slot != STASH_SIZE){
			grp.full_head = grp.stash[slot].next.adr;
			db_printf("FULL : %d << %d\n",slot,grp.full_head);
		}
		return slot;

	}


	/*
	// Inserts a full slot into the stash. This should only be called if there is enough space in
	// the link stash.
	*/
	 __device__ static void insert_full_slot(_CTX_ARGS, unsigned int slot){

		grp.stash[slot].next.adr = grp.full_head;
		db_printf("FULL : >> %d -> %d\n",slot,grp.full_head);
		grp.full_head = slot;

	}




	/*
	// Attempts to fill the link stash to the given threshold with links. This should only ever
	// be called in a single-threaded manner.
	*/
	 __device__ static void fill_stash_links(_CTX_ARGS, unsigned int threashold){


		unsigned int active = __activemask();
		__syncwarp(active);

		if( util::current_leader() ){

			for(int try_itr=0; try_itr < RETRY_LIMIT; try_itr++){
		
				if(grp.link_stash_count >= threashold){
					break;
				}
				/*	
				// Attempt to pull a queue from the pool. This should be very unlikely to fail unless
				// almost all links have been exhausted or the pool size is disproportionately small
				// relative to the number of work groups. In the worst case, this should simply not 
				// althdate any links, and the return value shall report this.
				*/
				unsigned int src_index = LinkAdrType::null;
				unsigned int start = util::random_uint(thd.rand_state)%POOL_SIZE;
				QueueType queue = pull_queue(glb.pool->queues,start,POOL_SIZE,src_index);
				q_printf("Pulled queue (%d,%d) from pool %d\n",get_head(queue),get_tail(queue),src_index);

				/*
				// Keep popping links from the queue until the full number of links have been added or
				// the queue runs out of links.
				*/
				for(int i=grp.link_stash_count; i < threashold; i++){
					LinkAdrType link = pop_front(_CTX_REFS,queue);
					if( ! link.is_null() ){
						insert_stash_link(_CTX_REFS,link);
						q_printf("Inserted link %d into link stash\n",link);
					} else {
						break;
					}
				}
				__threadfence();
				push_queue(_CTX_REFS,glb.pool->queues[src_index],queue);
				q_printf("Pushed queue (%d,%d) to pool %d\n",get_head(queue),get_tail(queue),src_index);

			}

		}
		__syncwarp(active);

	}




	/*
	// If the number of links in the link stash exceeds the given threshold value, this function frees
	// enough links to bring the number of links down to the threshold. This should only ever be
	// called in a single_threaded manner.
	*/
	 __device__ static void spill_stash_links(_CTX_ARGS, unsigned int threashold){

		/*
		// Do not even try if no links can be or need to be removed
		*/
		if(threashold >= grp.link_stash_count){
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
		unsigned int spill_count = grp.link_stash_count - threashold;
		for(unsigned int i=0; i < spill_count; i++){
			
			LinkAdrType link = claim_stash_link(_CTX_REFS);
			q_printf("Claimed link %d from link stash\n",link);
			push_back(_CTX_REFS,queue,link);

		}

		grp.link_stash_count = threashold;


		/*
		// Push out the queue to the pool
		*/
		q_printf("Pushing queue (%d,%d) to pool\n",get_head(queue),get_tail(queue));
		unsigned int dest_idx = util::random_uint(thd.rand_state) % POOL_SIZE;
		__threadfence();
		push_queue(_CTX_REFS,glb.pool->queues[dest_idx],queue);
		
		q_printf("Pushed queue (%d,%d) to pool\n",get_head(queue),get_tail(queue));

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
	 __device__ static void pop_frame_counters(_CTX_ARGS, unsigned int start_level, unsigned int end_level){


		unsigned int depth_dec = 0;
		unsigned int delta;
		unsigned int result;

		FrameType& frame = glb.stack->frames[start_level];

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
			FrameType& frame = glb.stack->frames[d];
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
			result = atomicSub(&(glb.stack->depth_live),depth_dec);
			if(result == 0){
				set_flags(grp,COMPLETION_FLAG);
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
	 __device__ static void push_promises(_CTX_ARGS, unsigned int level, unsigned int index, QueueType queue, int promise_delta) {


		LinkAdrType tail = queue.get_tail();
		LinkAdrType head = queue.get_head();
		rc_printf("SM %d: push_promises(level:%d,index:%d,queue:(%d,%d),delta:%d)\n",bthdkIdx.x,level,index,tail,head,promise_delta);
		/*
		// Do not bother pushing a null queue if there is no delta to report
		*/
		if( ( ! queue.is_null() ) || (promise_delta != 0) ){

			/*
			// Change the resident counter of the destination frame by the number of promises
			// that have been added to or removed from the given queue
			*/
			FrameType &dest = get_frame(_CTX_REFS,level);		
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


			grp.SM_promise_delta += promise_delta;
			rc_printf("SM %d-%d: Old count: %d, New count: %d\n",bthdkIdx.x,threadIdx.x,old_count,new_count);

			
			//rc_printf("SM %d-%d: frame zero resident count is: %d\n",bthdkIdx.x,threadIdx.x,glb.stack->frames[0].children_residents);
			/*
			// If the addition caused a frame to change from empty to non-empty or vice-versa,
			// make an appropriate incrementation or decrementation at the stack base.
			*/
			if( (old_count == 0) && (new_count != 0) ){
				atomicAdd(&(glb.stack->depth_live),0x00010000u);
			} else if( (old_count != 0) && (new_count == 0) ){
				atomicSub(&(glb.stack->depth_live),0x00010000u);
			} else {
				rc_printf("SM %d: No change!\n",bthdkIdx.x);
			}

			__threadfence();	
			/*
			// Finally, push the queues
			*/
			push_queue(_CTX_REFS,dest.pool.queues[index],queue);
			rc_printf("SM %d: Pushed queue (%d,%d) to stack at index %d\n",bthdkIdx.x,get_head(queue),get_tail(queue),index);
		
				
			//rc_printf("SM %d-%d: After queue pushed to stack, frame zero resident count is: %d\n",bthdkIdx.x,threadIdx.x,glb.stack->frames[0].children_residents);

		}
		rc_printf("(%d) the delta: %d\n",bthdkIdx.x,promise_delta);

	}



	/*
	// Attempts to pull a queue of promises from the frame in the stack of the given level, starting the 
	// pull attempt at the given index in the frame. If no queue could be pulled after attempting a 
	// pull at each queue in the given frame, a QueueType::null value is returned.
	*/
	 __device__ static QueueType pull_promises(_CTX_ARGS, unsigned int level, unsigned int& source_index) {


		rc_printf("SM %d: pull_promises(level:%d)\n",bthdkIdx.x,level);
		unsigned int src_idx = util::random_uint(thd.rand_state) % FRAME_SIZE;

		FrameType &src = get_frame(_CTX_REFS,level);
		
		QueueType queue = pull_queue(src.pool.queues,src_idx,FRAME_SIZE,source_index);
		if( ! queue.is_null() ){
			q_printf("SM %d: Pulled queue (%d,%d) from stack at index %d\n",bthdkIdx.x,get_tail(queue),get_head(queue), src_idx);
		} else {
			q_printf("SM %d: Failed to pull queue from stack starting at index %d\n",bthdkIdx.x, src_idx);
		}
		return queue;

	}


	/*
	// Attempts to pull a queue from any frame in the stack, starting from the highest and working
	// its way down. If no queue could be pulled, a QueueType::null value is returned.
	*/
	 __device__ static QueueType pull_promises_any_level(_CTX_ARGS, unsigned int& level, unsigned int& source_index){


		QueueType result;
		result.data = QueueType::null;
		unsigned int start_level = highest_level(_CTX_REFS);
		for(int level_itr = start_level; level_itr>=0; level_itr--){
			q_printf("Pulling promises at level %d for pull_promises_any_level\n",level_itr);
			result = pull_promises(_CTX_REFS,level_itr,source_index);
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
	 __device__ static LinkAdrType produce_link(_CTX_ARGS, unsigned int slot_index ){


		__shared__ LinkAdrType result;

		unsigned int active = __activemask();


		//__syncwarp(active);
		
		//if(util::current_leader()){
			LinkAdrType link_index = claim_stash_link(_CTX_REFS);
			db_printf("Claimed link %d from stash\n",link_index);
			LinkType& the_link = glb.arena[link_index];
			//grp.SM_promise_delta += grp.stash[slot_index].count;
			grp.stash[slot_index].next = LinkAdrType::null;
			the_link = grp.stash[slot_index];
			db_printf("Link has count %d and next %d in global memory",the_link.count, the_link.next);
			result = link_index;
			grp.stash_count -= 1;
		//}

		
		//__syncwarp(active);
		return result;

	}








	/*
	// Removes all promises in the stash that do not correspond to the given level, or to the levels
	// immediately above or below (level+1) and (level-1).
	*/
	 __device__ static void relevel_stash(_CTX_ARGS, unsigned int level){

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
	 __device__ static void spill_stash(_CTX_ARGS, unsigned int threashold){

		unsigned int active =__activemask();
		__syncwarp(active);


		
	#if DEF_STACK_MODE == 0


		if(util::current_leader()){
			db_printf("Current stash_count: %d\n",grp.stash_count);
		}
		
		
		if(util::current_leader() && (grp.stash_count > threashold)){

			unsigned int spill_count = grp.stash_count - threashold;
			int delta = 0;
			fill_stash_links(_CTX_REFS,spill_count);
			
			QueueType queue;
			queue.pair.data = QueueType::null;
			unsigned int partial_iter = 0;
			bool has_full_slots = true;
			for(unsigned int i=0; i < spill_count; i++){
				unsigned int slot = STASH_SIZE;
				if(has_full_slots){
					slot = claim_full_slot(_CTX_REFS);
					if(slot == STASH_SIZE){
						has_full_slots = false;
					}
				}
				if(! has_full_slots){
					for(;partial_iter < FN_ID_COUNT; partial_iter++){
						db_printf("%d",partial_iter);
						if(grp.partial_map[partial_iter] != STASH_SIZE){
							slot = grp.partial_map[partial_iter];
							partial_iter++;
							break;
						}
					}
				}
				if(slot == STASH_SIZE){
					break;
				}
				
				delta += grp.stash[slot].count;
				db_printf("Slot for production (%d) has %d promises\n",slot,grp.stash[slot].count);
				LinkAdrType link = produce_link(_CTX_REFS,slot);
				push_back(_CTX_REFS,queue,link);
				insert_empty_slot(_CTX_REFS,slot);
				if(grp.stash_count <= threashold){
					break;
				}
			}
		
			
			unsigned int push_index = util::random_uint(thd.rand_state)%FRAME_SIZE;
			rc_printf("Pushing promises for spilling\n");
			push_promises(_CTX_REFS,0,push_index,queue,delta);
			db_printf("Pushed queue (%d,%d) to stack\n",get_head(queue),get_tail(queue));
			
		
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
				unsigned int depth = grp.stash[i].depth;
				unsigned int size = grp.stash[i].size;
				unsigned int idx = (depth != level) ? 0 : 1;
				idx += (size >= WARP_COUNT) ? 0 : 2;
				bucket[idx] += 1;
			} 

			/*
			// Determine how much of which type of link needs to be dumped
			*/
			unsigned int dump_total = bucket[0];
			unsigned int dump_count = (grp.stash_count > threshold) ? grp.stash_count - threshold : 0;
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
			for(unsigned int i=0; i < grp.stash_count; i++){
				unsigned int depth = grp.stash[i].depth;
				unsigned int size  = grp.stash[i].size;
				unsigned int bucket_idx = (depth != level) ? 0 : 1;
				bucket_idx += (size >= WARP_COUNT) ? 0 : 2;
				if(bucket[bucket_idx] == 0){
					continue;
				}
				LinkAdrType link = grp.link_stash[grp.link_stash_count];
				grp.link_stash_count -= 1;

				copy_link(glb.arena[link], grp.stash[i]);

				unsigned int level_index = level+1-depth;
				counts[level_index] += size;
				push_back(_CTX_REFS,queues[level_index],link);

				grp.stash[i].size = 0;
			}
		}

	#endif
		
		__syncwarp(active);
		

	 
	}



	 __device__ static void async_call_stash_dump(_CTX_ARGS, Fn func_id, int depth_delta, unsigned int delta){

		/*
		// Make room to queue incoming promises, if there isn't enough room already.
		*/
		#if 0
		if(grp.link_stash_count < 1){
			fill_stash_links(_CTX_REFS,1);
		}

		if(grp.stash_count >= (STASH_SIZE-1)){
			spill_stash(_CTX_REFS, STASH_SIZE-2);
		}
		#else
		unsigned int depth = (unsigned int) (grp.level + depth_delta);
		unsigned int left_jump = partial_map_index(func_id,depth,grp.level);
		unsigned int space = 0;
		if( left_jump != PART_ENTRY_COUNT ){
			unsigned int left_idx = grp.partial_map[left_jump];	
			if( left_idx != STASH_SIZE ){
				space = WORK_GROUP_SIZE - grp.stash[left_idx].count;
			}
		}
		if( (grp.stash_count >= (STASH_SIZE-1)) && (space < delta) ){
			if(grp.link_stash_count < 1){
				fill_stash_links(_CTX_REFS,1);
			}
			spill_stash(_CTX_REFS, STASH_SIZE-2);
		}
		#endif

	}


	 __device__ static void async_call_stash_prep(_CTX_ARGS, Fn func_id, int depth_delta, unsigned int delta,
		unsigned int &left, unsigned int &left_start, unsigned int &right
	){

		/*
		// Locate the destination links in the stash that the promises will be written to. For now,
		// like many other parts of the code, this will be single-threaded within the work group
		// to make validation easier but will be optimized for group-level parallelism later.
		*/
		if( util::current_leader() ){

			db_printf("Queueing %d promises of type %d\n",delta,func_id);
			/*
			// Null out the right index. This index should not be used unless the number of
			// promises queued spills over beyond the first link being written to (the left one)
			*/
			right = STASH_SIZE;

			/*
			// Find the index of the partial link in the stash corresponding to the id and
			// depth of the calls being queued (if it exists).
			*/
			unsigned int depth = (unsigned int) (grp.level + depth_delta);
			unsigned int left_jump = partial_map_index(func_id,depth,grp.level);
			
			/*
			// If there is a partially filled link to be filled, assign that to the left index
			*/
			if(left_jump != PART_ENTRY_COUNT){
				//db_printf("A\n");
				left = grp.partial_map[left_jump];
			}

			unsigned int left_count;
			if(left == STASH_SIZE){
				//db_printf("B\n");
				left = claim_empty_slot(_CTX_REFS);
				grp.stash_count += 1;
				db_printf("Updated stash count: %d\n",grp.stash_count);
				grp.stash[left].id    = func_id;
				grp.partial_map[left_jump] = left;
				left_count = 0;
			} else {
				left_count = grp.stash[left].count;
			}

			if ( (left_count + delta) > WORK_GROUP_SIZE ){
				//db_printf("C\n");
				right = claim_empty_slot(_CTX_REFS);
				grp.stash_count += 1;
				db_printf("Updated stash count: %d\n",grp.stash_count);
				grp.stash[right].count = left_count+delta - WORK_GROUP_SIZE;
				grp.stash[right].id    = func_id;
				insert_full_slot(_CTX_REFS,left);
				grp.partial_map[left_jump] = right;
				grp.stash[left].count = WORK_GROUP_SIZE;
			} else if ( (left_count + delta) == WORK_GROUP_SIZE ){
				//db_printf("D\n");
				grp.partial_map[left_jump] = STASH_SIZE;
				insert_full_slot(_CTX_REFS,left);
				grp.stash[left].count = WORK_GROUP_SIZE;
			} else {
				grp.stash[left].count = left_count + delta;
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
	 __device__ static void async_call(_CTX_ARGS, Fn func_id, int depth_delta, PromiseUnionType promise){

		unsigned int active = __activemask();

		/*
		// Calculate how many promises are being queued as well as the assigned index of the 
		// current thread's promise in the write to the stash.
		*/
		unsigned int index = util::warp_inc_scan();
		unsigned int delta = util::active_count();

		async_call_stash_dump(_CTX_REFS, func_id, depth_delta, delta);

		__shared__ unsigned int left, left_start, right;


		async_call_stash_prep(_CTX_REFS,func_id,depth_delta,delta,left,left_start,right);
		
		/*
		// Write the promise into the appropriate part of the stash, writing into the left link 
		// when possible and spilling over into the right link when necessary.
		*/
		__syncwarp(active);
		if( (left_start + index) >= WORK_GROUP_SIZE ){
			//db_printf("Overflow: id: %d, left: %d, left_start: %d, index: %d\n",threadIdx.x,left,left_start,index);
			grp.stash[right].promises[left_start+index-WORK_GROUP_SIZE] = promise;
		} else {
			//db_printf("Non-overflow: id: %d, left: %d, left_start: %d, index: %d\n",threadIdx.x,left,left_start,index);
			grp.stash[left].promises[left_start+index] = promise;
		}
		__syncwarp(active);	

	}


	/*
	// Like async_call, but allows for one to hand in the underlying type corresponding to a function id directly
	*/
	template<Fn FUNC_ID>
	 __device__ static void async_call_cast(_CTX_ARGS, int depth_delta, typename PromiseType<FUNC_ID>::ParamType param_value){

		unsigned int active = __activemask();

		/*
		// Calculate how many promises are being queued as well as the assigned index of the 
		// current thread's promise in the write to the stash.
		*/
		unsigned int index = util::warp_inc_scan();
		unsigned int delta = util::active_count();

		async_call_stash_dump(_CTX_REFS, FUNC_ID, depth_delta, delta);

		__shared__ unsigned int left, left_start, right;


		async_call_stash_prep(_CTX_REFS,FUNC_ID,depth_delta,delta,left,left_start,right);
		
		/*
		// Write the promise into the appropriate part of the stash, writing into the left link 
		// when possible and spilling over into the right link when necessary.
		*/
		__syncwarp(active);
		if( (left_start + index) >= WORK_GROUP_SIZE ){
			//db_printf("Overflow: id: %d, left: %d, left_start: %d, index: %d\n",threadIdx.x,left,left_start,index);
			grp.stash[right].promises[left_start+index-WORK_GROUP_SIZE].template cast<FUNC_ID>() = param_value;
		} else {
			//db_printf("Non-overflow: id: %d, left: %d, left_start: %d, index: %d\n",threadIdx.x,left,left_start,index);
			grp.stash[left].promises[left_start+index].template cast<FUNC_ID>() = param_value;
		}
		__syncwarp(active);	

	}


	//#define PARACON

	/*
	// Adds the contents of the link at the given index to the stash and adds the given link to link
	// stash. Once complete, it returns the number of promises added to the stash by the operation.
	// This should only ever be called if there is enough space to store the extra work and link.
	*/
	 __device__ static unsigned int consume_link(_CTX_ARGS, LinkAdrType link_index ){


		#ifdef PARACON
		__shared__ LinkAdrType the_index;
		__shared__ unsigned int add_count;
		__shared__ unsigned int func_id;

		unsigned int active = __activemask();


		__syncwarp(active);
		
		if(util::current_leader()){

			the_index = link_index;
			add_count = glb.arena[link_index].count;
			func_id   = glb.arena[link_index].id;

		}

		
		__syncwarp(active);
		
		if(threadIdx.x < add_count){
			PromiseType promise = glb.arena[the_index].data.data[threadIdx.x];
			async_call(_CTX_REFS,func_id,0,promise);
		}


		__syncwarp(active);


		if(util::current_leader()){
			insert_stash_link(_CTX_REFS,link_index);
		}

		return add_count;


		#else 

		LinkAdrType the_index;
		unsigned int add_count;
		Fn           func_id;

		unsigned int active = __activemask();
		unsigned int acount = util::active_count();

		

		the_index = link_index;
		add_count = glb.arena[link_index].count;
		func_id   = glb.arena[link_index].id;

		//grp.SM_promise_delta -= add_count;
		
		db_printf("active count: %d, add count: %d\n",acount,add_count);

		
		db_printf("\n\nprior stash count: %d\n\n\n",grp.stash_count);
		//*
		for(unsigned int i=0; i< add_count; i++){
			PromiseUnionType promise = glb.arena[the_index].promises[i];
			async_call(_CTX_REFS,func_id,0,promise);
		}
		// */
		//PromiseType promise = glb.arena[the_index].data.data[0];
		//async_call(_CTX_REFS,func_id,0,promise);

		db_printf("\n\nafter stash count: %d\n\n\n",grp.stash_count);


		insert_stash_link(_CTX_REFS,link_index);

		return add_count;



		#endif


	}






	/*
	// Tries to transfer links from the stack into the stash of the work group until the stash
	// is filled to the given threashold. If a halting condition is reached, this function will set
	// the keep_running value in the shared context to false.
	*/
	 __device__ static void fill_stash(_CTX_ARGS, unsigned int threashold){

		unsigned int active =__activemask();
		__syncwarp(active);
		

		#ifdef PARACON
		__shared__ unsigned int link_count;
		__shared__ unsigned int links[STASH_SIZE];
		#endif

		//db_printf("\n\n\nXXXXX\n\n\n");
		/*
		// Currently implemented in a single-threaded manner per work group to simplify the initial
		// correctness checking process. This can later be changed to take advantage of in-group
		// parallelism.
		*/
		if(util::current_leader()){

			//db_printf("Filling stash...\n");

			unsigned int taken = 0;
			
			threashold = (threashold > STASH_SIZE) ? STASH_SIZE : threashold;
			unsigned int gather_count = (threashold < grp.stash_count) ? 0  : threashold - grp.stash_count;
			if( (STASH_SIZE - grp.link_stash_count) < gather_count){
				unsigned int spill_thresh = STASH_SIZE - gather_count;
				spill_stash_links(_CTX_REFS,spill_thresh);
			}
			

			#ifdef PARACON
			unsigned int thd_link_count = 0;
			#endif

			#ifdef RACE_COND_PRINT
			unsigned int p_depth_live = glb.stack->depth_live;
			rc_printf("SM %d: depth_live is (%d,%d)\n",bthdkIdx.x,left_half(p_depth_live),right_half(p_depth_live));
			#endif

			for(unsigned int i = 0; i < RETRY_LIMIT; i++){

				/* If the stack is empty or a flag is set, return false */
				unsigned int depth_live = glb.stack->depth_live;
				if( (depth_live == 0u) || ( glb.stack->event_com != 0u) ){
					grp.keep_running = false;
					break;
				}


				unsigned int src_index;
				QueueType queue;

				#if DEF_STACK_MODE == 0
			
				db_printf("STACK MODE ZERO\n");	
				q_printf("%dth try pulling promises for fill\n",i+1);
				queue = pull_promises(_CTX_REFS,grp.level,src_index);

				#else
				/*
				// Determine whether or not to pull from the current level in the stack
				*/
				unsigned int depth = left_half(depth_live);
				bool pull_any = (depth < grp.level);
				FrameType &current_frame = get_frame(_CTX_REFS,depth);
				if(!pull_any){
					pull_any = (right_half(current_frame.children_residents) == 0);
				}


				/*
				// Retrieve a queue from the stack.
				*/

				if(pull_any){
					unsigned int new_level;
					queue = pull_promises_any_level(_CTX_REFS,new_level,src_index);
					relevel_stash(_CTX_REFS,new_level);
				} else {
					queue = pull_promises(_CTX_REFS,grp.level,src_index);
				}
				#endif

				#ifdef PARACON
				db_printf("About to pop promises\n");
				while(	( ! queue.is_null() ) 
				     && (thd_link_count < gather_count)
				     && (grp.link_stash_count < STASH_SIZE) 
				){
					LinkAdrType link = pop_front(_CTX_REFS,queue);					
					if(link != LinkAdrType::null){
						db_printf("Popping front %d\n",link);
						links[thd_link_count] = link;
						taken += glb.arena[link].count;
						thd_link_count++;
					} else {
						break;
					}
				}
				#else
				db_printf("About to pop promises\n");
				while(	( ! queue.is_null() ) 
				     && (grp.stash_count < threashold)
				     && (grp.link_stash_count < STASH_SIZE) 
				){
					LinkAdrType link = pop_front(_CTX_REFS,queue);					
					if( ! link.is_null() ){
						q_printf("Popping front %d\n",link);
						taken += consume_link(_CTX_REFS,link);
					} else {
						break;
					}
				}
				#endif
		
				db_printf("Popped promises\n");
				if(taken != 0){
					if(!grp.busy){
						atomicAdd(&(glb.stack->depth_live),1);
						grp.busy = true;
						rc_printf("SM %d: Incremented depth value\n",bthdkIdx.x);
					}
					rc_printf("Pushing promises for filling\n");	
					push_promises(_CTX_REFS,grp.level,src_index,queue,-taken);
					break;
				}
		
				#ifndef PARACON
				if( grp.stash_count >= threashold ){
					break;
				}
				#endif

			}
		




			if(grp.busy && (grp.stash_count == 0)){
				atomicSub(&(glb.stack->depth_live),1);
				rc_printf("SM %d: Decremented depth value\n",bthdkIdx.x);
				grp.busy = false;
			}


			#ifdef PARACON
			link_count = thd_link_count;
			#endif

			
		}

		__syncwarp(active);



		#ifdef PARACON

		for(int i=0; i<link_count;i++){
			consume_link(_CTX_REFS,links[i]);
		}


		__syncwarp(active);
		#endif



	}




	 __device__ static void clear_exec_head(_CTX_ARGS){

		
		if( util::current_leader() && (grp.exec_head != STASH_SIZE) ){
			insert_empty_slot(_CTX_REFS,grp.exec_head);
			grp.exec_head = STASH_SIZE;
		}
		__syncwarp();

	}




	/*
	// Selects the next link in the stash. This selection process could become more sophisticated
	// in later version to account for the average branching factor of each async function. For now,
	// it selects the fullest slot of the current level if it can. If no slots with promises for the
	// current level exist in the stash, the function returns false.
	*/
	 __device__ static bool advance_stash_iter(_CTX_ARGS){

		__shared__ bool result;
		unsigned int active =__activemask();
		__syncwarp(active);
		

		if(util::current_leader()){

			if(grp.full_head != STASH_SIZE){
				grp.exec_head = claim_full_slot(_CTX_REFS);
				grp.stash_count -= 1;
				result = true;
				//db_printf("Found full slot.\n");
			} else {
				//db_printf("Looking for partial slot...\n");
				unsigned int best_id   = PART_ENTRY_COUNT;
				unsigned int best_slot = STASH_SIZE;
				unsigned int best_count = 0;
				for(int i=0; i < FN_ID_COUNT; i++){
					unsigned int slot = grp.partial_map[i];
					
					if( (slot != STASH_SIZE) && (grp.stash[slot].count > best_count)){
						best_id = i;
						best_slot = slot;
						best_count = grp.stash[slot].count;
					}
					
				}

				result = (best_slot != STASH_SIZE);
				if(result){
					//db_printf("Found partial slot.\n");
					grp.exec_head = best_slot;
					grp.partial_map[best_id] = STASH_SIZE;
					grp.stash_count -=1;
				}
			}

		}

		__syncwarp(active);
		return result;

	}




	/*
	// Tries to perform up to one work group worth of work by selecting a link from shared memory (or,
	// if necessary, fetching a link from global memory), and running the function on the data within
	// the link, as directed by the function id the link is labeled with. This function returns false
	// if a halting condition has been reached (either due to lack of work or an event) and true
	// otherwise.
	*/
	 __device__ static void exec_cycle(_CTX_ARGS){


		clear_exec_head(_CTX_REFS);

		/*
		// Advance the stash iterator to the next chunk of work that needs to be done.
		*/
		//*
		if( grp.can_make_work && (grp.full_head == STASH_SIZE) ){
			grp.can_make_work = make_work(_CTX_REFS,_STATE_REFS);
			if( util::current_leader() && (! grp.busy ) && ( grp.stash_count != 0 ) ){
				atomicAdd(&(glb.stack->depth_live),1);
				grp.busy = true;
			}
		}


		#if 1
		if(grp.full_head == STASH_SIZE){
			fill_stash(_CTX_REFS,STASH_SIZE-2);
		}
		#else
		if(grp.full_head == STASH_SIZE){
			if( !grp.scarce_work ){
				fill_stash(_CTX_REFS,STASH_SIZE-2);
				if( util::current_leader() && (grp.full_head == STASH_SIZE) ){
					grp.scarce_work = true;
				}
			}
		} else {
			if( util::current_leader() ){
				grp.scarce_work = false;
			}
		}
		#endif
		// */

		if( !advance_stash_iter(_CTX_REFS) ){
			/*
			// No more work exists in the stash, so try to fetch it from the stack.
			*/
			fill_stash(_CTX_REFS,STASH_SIZE-2);

			if( grp.keep_running && !advance_stash_iter(_CTX_REFS) ){
				/*
				// REALLY BAD: The fill_stash function successfully, however 
				// the stash still has no work to perform. In this situation,
				// we set an error flag and halt.
				*/
				/*
				if(util::current_leader()){
					db_printf("\nBad stuff afoot!\n\n");
				}
				set_flags(grp,STASH_FAIL_FLAG);
				grp.keep_running = false;
				*/
			}
		}

		
		unsigned int active = __activemask();
		__syncwarp(active);


		if( (grp.exec_head != STASH_SIZE) && grp.keep_running ){
			/* 
			// Find which function the current link corresponds to.
			*/	
			Fn func_id     = grp.stash[grp.exec_head].id;
			unsigned int promise_count = grp.stash[grp.exec_head].count;
			
			/*
			// Only execute if there is a promise in the current link corresponding to the thread that
			// is being executed.
			*/
			if(util::current_leader()){
				q_printf("Executing slot %d, which is %d promises of type %d\n",grp.exec_head,promise_count,func_id);
			}
			if( threadIdx.x < promise_count ){
				//db_printf("Executing...\n");
				PromiseUnionType& promise = grp.stash[grp.exec_head].promises[threadIdx.x];
				//do_async(_CTX_REFS,func_id,promise);
				promise.template loose_eval<ProgramType>(func_id,_CTX_REFS,glb.global_state,grp.group_state,thd.thread_state);
			}
		}

		__syncwarp(active);


	}



	 __device__ static void cleanup_runtime(_CTX_ARGS){

		unsigned int active = __activemask();
		__syncwarp(active);
		if(util::current_leader()){
			q_printf("CLEANING UP\n");
			spill_stash(_CTX_REFS,0);
			spill_stash_links(_CTX_REFS,0);	
			if(grp.busy){
				atomicSub(&(glb.stack->depth_live),1);
			}

		}




	}


	/*
	//
	// This must be run once on the resources used for execution, prior to execution. Given that this
	// essentially wipes all data from these resources and zeros all values, it is not advised that
	// this function be used at any other time, except to setup for a re-start or to clear out after
	// calling the pull_runtime to prevent promise duplication.
	//
	*/
	 __device__ static void init(
		LinkType*  __restrict__ arena, size_t arena_size,
		PoolType*  __restrict__ pool,
		StackType* __restrict__ stack
	){



		/* Initialize per-thread resources */
		ThreadContext thd = init_thread();


		const unsigned int threads_per_frame = FRAME_SIZE + 1;
		const unsigned int total_stack_work = StackType::NULL_LEVEL * threads_per_frame;
		
		unsigned int worker_count = gridDim.x*blockDim.x;

		/*
		// If the currently executing thread has global thread index 0, wipe the data in the base
		// of the stack.
		*/
		if(thd.thread_id == 0){
			stack->event_com  = 0;
			stack->depth_live = 0;
		}


		/*
		if( thd.thread_id == 0 ){
			printf(	"Initializing the stack with\n"
				"\t- total_stack_work=%d\n"
				"\t- threads_per_frame=%d\n"
				"\t- worker_count=%d\n"
				"\t- stack->frames[0].children_residents=%d\n",
				total_stack_work,
				threads_per_frame,
				worker_count,
				stack->frames[0].children_residents
			);
		}
		*/



		/*
		// Blank out the frames in the stack. Setting queues to NULL_QUEUE, and zeroing the counts
		// for resident promises and child promises of each frame.
		*/
		for(unsigned int index = thd.thread_id; index < total_stack_work; index+=worker_count ){
			
			unsigned int target_level = index / threads_per_frame;
			unsigned int frame_index  = index % threads_per_frame;
			if( frame_index == FRAME_SIZE ){
				stack->frames[target_level].children_residents = 0u;
			} else {
				stack->frames[target_level].pool.queues[threadIdx.x].pair.data = QueueType::null;
			}

		}


		/*
		// Initialize the arena, connecting the contained links into roughly equally sized lists,
		// zeroing the promise counter in the links and marking the function ID with an invalid
		// value to make use-before-initialization more obvious during system validation.
		*/
		unsigned int bump = ((arena_size%POOL_SIZE) != 0) ? 1 : 0;
		unsigned int arena_init_stride = arena_size/POOL_SIZE + bump;
		for(unsigned int index = thd.thread_id; index < arena_size; index+=worker_count ){
			
			unsigned int next = index + 1;
			if( ( (next % arena_init_stride) == 0 ) || (next >= arena_size) ){
				next = LinkAdrType::null;
			}
			arena[index].empty(LinkAdrType(next));

		}


		/*
		// Initialize the pool, giving each queue slot one of the previously created linked lists.
		*/
		for(unsigned int index = thd.thread_id; index < POOL_SIZE; index+=worker_count ){	
			
			unsigned int head = arena_init_stride * index;
			unsigned int tail = arena_init_stride * (index + 1) - 1;
			tail = (tail >= arena_size) ? arena_size - 1 : tail;
			pool->queues[index] = QueueType(LinkAdrType(head),LinkAdrType(tail));

		}


	}


	/*
	// Unpacks all promise data from the call buffer into the stack of the given context. This
	// could be useful for backing up program states for debugging or to re-start processing from
	// a previous state.
	*/
	 __device__ static void push_calls(GlobalContext glb, LinkType* call_buffer, size_t link_count){


		/* Initialize per-warp resources */
		__shared__ GroupContext grp;
		init_group(glb,grp);
		
		/* Initialize per-thread resources */
		ThreadContext thd;
		init_local(thd);	


		for(int link_index=blockIdx.x; link_index < link_count; link_index+= gridDim.x){
			LinkType& the_link = call_buffer[link_index];
			unsigned int count   = the_link.count;
			unsigned int func_id = the_link.id;
			if(threadIdx.x < count){
				db_printf("\nasync_call(id:%d,depth: 0)\n\n",func_id);
				async_call(_CTX_REFS,func_id,0,the_link.data.data[threadIdx.x]);
			}

		}

		cleanup_runtime(_CTX_REFS);

	}




	static void checkError(){

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

		checkError();
		
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
	 __device__ static void pull_promises(Instance &instance){

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
	 __device__ static void exec(GlobalContext glb, unsigned int cycle_count){

		

		/* Initialize per-warp resources */
		__shared__ GroupContext grp;
		init_group(grp);
		
		/* Initialize per-thread resources */
		ThreadContext thd = init_thread();

		initialize(_CTX_REFS,_STATE_REFS);

		if(util::current_leader()){
			rc_printf("\n\n\nInitial frame zero resident count is: %d\n\n\n",runtime.stack->frames[0].children_residents);
		}	

		/* The execution loop. */
		#ifdef RACE_COND_PRINT
		unsigned int cycle_break = cycle_count;
		#endif
		for(unsigned int cycle=0u; cycle<cycle_count; cycle++){
			/* Early halting handled with a break. */
			exec_cycle(_CTX_REFS);
			if(!grp.keep_running){
				#ifdef RACE_COND_PRINT
				cycle_break = cycle+1;
				#endif
				break;
			}		
		}

		finalize(_CTX_REFS,_STATE_REFS);

		/*
		// Ensure that nothing which should persist between dispatches is lost in the
		// shared or private memory of the halting program.
		*/
		cleanup_runtime(_CTX_REFS);

			
		if(util::current_leader()){
			rc_printf("SM %d finished after %d cycles with promise delta %d\n",bthdkIdx.x,cycle_break,grp.SM_promise_delta);
		}

	}


};



/*
// These functions are here just to trampoline into the actual main functions for a given program.
// This is done because structs/classes may not have global member functions.
*/
template<typename ProgType>
__global__ void _dev_init(typename ProgType::GlobalContext glb) {
	ProgType::init(
		glb.arena.links,glb.arena.size,
		glb.pool,
		glb.stack
	);
}


template<typename ProgType>
__global__ void _dev_exec(typename ProgType::GlobalContext global_context, unsigned int cycle_count) {
	ProgType::exec(global_context,cycle_count);
}




/*
// These functions unwrap an instance into its global context and passes it to the responsible
// kernel.
*/
template<typename ProgType>
__host__ void init(typename ProgType::Instance& instance,size_t group_count) {
	_dev_init<ProgType><<<group_count,ProgType::WORK_GROUP_SIZE>>>(instance.to_context());
}
template<typename ProgType>
__host__ void exec(typename ProgType::Instance& instance,size_t group_count, unsigned int cycle_count) {
	_dev_exec<ProgType><<<group_count,ProgType::WORK_GROUP_SIZE>>>(instance.to_context(),cycle_count);
}










#define DEF_PROMISE_TYPE(id, paramtype)                     \
        template<> struct PromiseType<id>                   \
        {                                                   \
                typedef paramtype ParamType;                \
        };                                                  \


#define PARAM_TYPE(id) typename PromiseType<id>::ParamType


#define HARM_ARG_PREAMBLE(progtype)                         \
                progtype::GlobalContext & _global_context,  \
                progtype::GroupContext  & _group_context,   \
                progtype::ThreadContext & _thread_context,  \
                progtype::GlobalState & global,             \
                progtype::GroupState  & group,              \
                progtype::ThreadState & thread              \

#define DEF_ASYNC_FN(progtype, id, arg_name)                \
        template<>  __device__                              \
        void promise_eval<progtype,id> (                    \
                HARM_ARG_PREAMBLE(progtype),                \
                PARAM_TYPE(id) arg_name                     \
        )                                                   \


#define DEF_INITIALIZE(progtype)                            \
        template<>  __device__                              \
        void progtype::initialize(                          \
                HARM_ARG_PREAMBLE(progtype)                 \
        )                                                   \

#define DEF_FINALIZE(progtype)                              \
        template<>  __device__                              \
        void progtype::finalize(                            \
                HARM_ARG_PREAMBLE(progtype)                 \
        )                                                   \

#define DEF_MAKE_WORK(progtype)                             \
        template<>  __device__                              \
        bool progtype::make_work(                           \
                HARM_ARG_PREAMBLE(progtype)                 \
        )                                                   \


#define ASYNC_CALL(fn_id,param) std::remove_reference<decltype(_global_context)>::type::ParentProgramType::template async_call_cast<fn_id>(_global_context,_group_context,_thread_context, 0, param)


