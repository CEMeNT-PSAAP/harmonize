
#include <stdio.h>
#include "math.h"

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




/*
// These values should eventually be externally defined by the build system so that these 
// values may be tailored to machines at compile time.
*/
#ifndef DEF_WG_COUNT
	#define DEF_WG_COUNT 			1
#endif

#ifndef DEF_WG_SIZE
	#define DEF_WG_SIZE 			32
#endif

#ifndef DEF_FUNCTION_ID_COUNT
	#define DEF_FUNCTION_ID_COUNT 		4
#endif

#ifndef DEF_THUNK_SIZE
	#define DEF_THUNK_SIZE 			4
#endif

#ifndef DEF_STASH_SIZE
	#define DEF_STASH_SIZE 			8
#endif

#ifndef DEF_STACK_MODE
	#define DEF_STACK_MODE			0
#endif

#ifndef DEF_RETRY_LIMIT
	//#define DEF_RETRY_LIMIT			64
	//#define DEF_RETRY_LIMIT			32
	#define DEF_RETRY_LIMIT			16
#endif

#ifndef DEF_QUEUE_WIDTH
	#define DEF_QUEUE_WIDTH			32
#endif



/*
// During system verification/debugging, this will be used as a cutoff to prevent infinite looping
*/
const unsigned int RETRY_LIMIT = DEF_RETRY_LIMIT;


/*
// This helps in determining how many warps and threads can reasonably be running simultaneously.
*/
const unsigned int WG_COUNT	= DEF_WG_COUNT;
const unsigned int WG_SIZE	= DEF_WG_SIZE;
const unsigned int WORKER_COUNT = WG_COUNT * WG_SIZE;


/*
// FUNCTION_ID_COUNT corresponds to the number of valid function IDs in the system. These function
// IDs should be in the range [0,FUNCTION_ID_COUNT) in order for the code to work. THUNK_SIZE
// corresponds to how many 32-bit fields of data are passed as part of a thunk. These factor
// into the runtime and memory footprint for work groups, and should be as small as possible.
*/
const unsigned int FUNCTION_ID_COUNT    = DEF_FUNCTION_ID_COUNT;
const unsigned int THUNK_SIZE 		= DEF_THUNK_SIZE;


/*
// How many queues can be stored per frame in the stack. Generally speaking, increasing this number
// will reduce the number of atomic operation conflicts in the stack. However, as the number of
// queues increases, the memory footprint of the stack increases and the operations required to
// find work when it is scarce increases as well.
*/
const unsigned int QUEUES_PER_FRAME = 32u;

/*
// The number of levels in the stack
*/
#if DEF_STACK_MODE == 0
const unsigned int STACK_SIZE   = 1u;
#else
const unsigned int STACK_SIZE	= 1024u;
#endif

/*
// How many slots for queues exist in the link pool. As with the stack, increasing this number
// can reduce atomic operation contention but can increase memory footprint as well as exacerbate
// search times during periods of high work scarcity.
*/
const unsigned int POOL_SIZE	= WG_COUNT;

/*
// The number of links that are stored in the shared memory for work caching and load balancing
// by work groups. It is strongly advised that this value be at least one more than the number
// of function ids for the current application. Due to the pidgeon hole principle, this means that
// you cannot run out of space in the cache without having at least one full link, which can
// be queued on the stack without offloading redistibution work on some future work group's 
// processing cycle. Ideally, a few more links than that should be available, as it is wise to
// have at least a few full links around to prevent idling in the work groups.
*/
const unsigned short STASH_SIZE	= DEF_STASH_SIZE;

/*
// The number of global-memory link references that a work group can hold in shared memory. This
// is necessary to guard against thrashing in the link pool due to sequential waves of thunk
// queueing and work starvation.
*/
// [TODO: Seperate stash size from link stash size]
//const unsigned int LINK_STASH_SIZE = 16u;

/*
// A null value used for the indexing space into the partial link maps of work groups. Partial link
// maps point to the partial link of a given function id / call depth combination for thunks.
*/
#if DEF_STACK_MODE == 0
const unsigned int PART_MAP_NULL = STASH_SIZE*3;
#elif DEF_STACK_MODE == 1
const unsigned int PART_MAP_NULL = STASH_SIZE;
/*
// A null value for the stack indexing space
*/
const unsigned short NULL_LEVEL	= 0xFFFFu;
/*
// The maximum level in the stack. In the production verision of this system, trying to queue
// thunks at depths abouve this level will result in early halting with an error.
*/
const unsigned int MAX_LEVEL	= STACK_SIZE-1;
#else
#error Invalid STACK_MODE value given. Valid values are 0 (do not use stack) and 1 (use stack).
#endif


/*
// NULL_LINK:  A null value for the link indexing space.
//
// NULL_QUEUE: A value corresponding to an empty queue.
//
// ARENA_SIZE: The number of links in the system.
//
// ctx_queue: The struct representation of queues on the stack. Should modifications need to be
// made to further annotate queues, it can be done through this struct, rather than through tedius
// find-and-replace code revision.
//
*/
#if DEF_QUEUE_WIDTH == 32
typedef unsigned short		ctx_link_adr;
typedef unsigned int		ctx_queue_t;

const ctx_link_adr	NULL_LINK	= 0xFFFFu;
const ctx_queue_t	NULL_QUEUE	= 0xFFFFFFFFu;
const ctx_queue_t	LINK_MASK	= 0x0000FFFFu;
const unsigned int	LINK_BITS	= 16u;

#ifndef DEF_ARENA_SIZE
const unsigned int ARENA_SIZE	= 0xFFFFu;
#elif DEF_ARENA_SIZE <= 0xFFFFu
const unsigned int ARENA_SIZE   = DEF_ARENA_SIZE;
#else
#error Invalid ARENA_SIZE value given. Valid values are less than or equal to 0xFFFFu for 32-bit queue widths
#endif

struct ctx_queue {
	ctx_queue_t data;
};

struct ctx_padded_queue {
	ctx_queue queue;
	//unsigned int padding[7];
};

#elif DEF_QUEUE_WIDTH == 64
typedef unsigned int		ctx_link_adr;
typedef unsigned long long int	ctx_queue_t;

const ctx_link_adr	NULL_LINK	= 0xFFFFFFFFu;
const ctx_queue_t	NULL_QUEUE	= 0xFFFFFFFFFFFFFFFFu;
const ctx_queue_t	LINK_MASK	= 0x00000000FFFFFFFFu;
const unsigned int	LINK_BITS	= 32u;

#ifndef DEF_ARENA_SIZE
const unsigned int ARENA_SIZE	= 0xFFFFFFFFu;
#elif DEF_ARENA_SIZE <= 0xFFFFFFFFu
const unsigned int ARENA_SIZE   = DEF_ARENA_SIZE;
#else
#error Invalid ARENA_SIZE value given. Valid values are less than or equal to 0xFFFFFFFFu for 64-bit queue widths
#endif

struct ctx_queue {
	ctx_queue_t data;
};

struct ctx_padded_queue {
	ctx_queue queue;
	//unsigned long long int padding[3];
};

#else
#error Invalid QUEUE_WIDTH value given. Valid values are 32 and 64.
#endif



/*
// The flags used to indicate the halting conditions that the system reaches. This is currently
// used mainly to detect invalid states when debugging, however it could potentially be used
// later to communicate other useful information to the CPU.
*/
const unsigned int BAD_FUNC_ID_FLAG	= 0x00000001;
const unsigned int STASH_FAIL_FLAG	= 0x00000002;
const unsigned int COMPLETION_FLAG	= 0x80000000;


/*
// This is the struct used to represent asynchronous function calls. In later iterations of this
// system, it will become more sophisticated. For now, it is simply an array of 32-bit values.
*/
struct ctx_thunk {
	unsigned int data[THUNK_SIZE];
};

/*
// This is the array of thunks that occupies the links held within the stack of the runtime context
// and within the shared memory of groups.
*/
struct ctx_link_payload {
	ctx_thunk data[WG_SIZE];
};






/*
// Stores links with thunks corresponding to a specific level in the runtime stack.
//
// To ensure that the depth of the stack is tracked correctly, the children_residents counter is 
// atomically modified to track the number of thunks in the frame as well as the number of direct
// child thunks present in the frame above.
*/
struct ctx_frame {
	
	unsigned int		children_residents;
	ctx_padded_queue	queues[QUEUES_PER_FRAME];

};


/*
// Stores a number of frames up until the maximum stack depth. This allows for thunks to be
// divided into groups based off of how deep into call recursion they are.
//
// The depth_live counter atomically tracks the height of the stack as well as the number of work 
// groups actively processing. Should this counter ever have a depth count and a live group count 
// of zero at the same time, no work can be performed, and so the program exits.
//
// The event_com counter has bits set to communicate events between work groups. Should a halting
// condition be reached, it would be communicated through this field.
*/
struct ctx_stack {

	unsigned int	event_com;
	unsigned int	depth_live;
	ctx_frame	frames[STACK_SIZE];

};



/*
// Reserved for future use for annotating links
*/
struct ctx_link_meta_data {
	unsigned int data;
};


/*
// Stores up to WG_SIZE thunks in the data field, all corresponding to the same function id, 
// which is stored in the id field, and all corresponding to the same call depth, stored in the
// depth field.
//
// When stored in a linked list as part of a queue, the next field is used to point to the next
// link in the list.
//
// The meta_data field is currently unused, but has promise for use in the addition of functional
// programming utilities later on in development.
*/
struct ctx_link {

	ctx_link_payload	data;
	ctx_link_adr		next;
	ctx_link_meta_data	meta_data;
	unsigned short		depth;
	unsigned short		id;
	unsigned short		count;

};


/*
// The local (or private, in OpenCL parlance) context used by individual threads. The thread_id
// is simply a unique identifier corresponding to its global x index. The rand_state field is
// used as a state for a simple random number generator needed for stochastic load balancing.
*/
struct ctx_local {

	unsigned int	thread_id;	
	unsigned int	rand_state;

};


/*
// The shared (or local, in OpenCL parlance) context used by work groups.
//
// - The level field corresponds to the current depth in the stack that thunks are being executed at
// - The stash_count field corresponds to the number of non-empty links in the stash.
// - The stash_iter field is an iterator over the stash used to indicate which link's thunks are 
//   currently being executed.
// - The link_stash_count is the number of link indexes allocated and cached by the warp for use
//   in queuing locally stored link data into the stack
// - The partial map field is an array of indexes pointing to which links in the stash are
//   partially filled with thunks for a given level and function id
// - The arena field points at the arena in global memory, which is indexed into to access links
// - The pool field points at the link pool, which is used to allocate links
// - The stack field points at the stack, which organizes links in global memory that are currently
//   full of thunks
// - The stash field stores links locally for thunk storage.
// - The link_stash field stores indexes to links in the arena for offloading thunk data elsewhere
*/
struct ctx_shared {

	unsigned short			level;		// Current level being run
	
	bool				keep_running;
	bool				busy;
	bool				can_make_work;	
	bool				scarce_work;	

	unsigned char			stash_count;	// Number of filled blocks in stash
	unsigned char			exec_head;	// Indexes the link that is/will be evaluated next
	unsigned char			full_head;	// Head of the linked list of full links
	unsigned char			empty_head;	// Head of the linked list of empty links

#if DEF_STACK_MODE == 0
	unsigned char			partial_map[FUNCTION_ID_COUNT]; // Points to partial blocks
#else
	unsigned char			partial_map[FUNCTION_ID_COUNT*3]; // Points to partial blocks
#endif

	unsigned char			link_stash_count; // Number of global-space links stored

	ctx_link* 		__restrict__	arena;
	ctx_padded_queue*	__restrict__	pool;
	ctx_stack* 		__restrict__	stack;
	ctx_link			stash[STASH_SIZE];
	ctx_link_adr			link_stash[STASH_SIZE];
	
	
	int				SM_thunk_delta;
	unsigned long long int		work_iterator;

};



///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
// Start of packing helper functions //////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions for pairs of 16 bit values packed into 32 bit fields //////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
const unsigned int MASK_16 = 0xFFFFu;
const unsigned int LEFT_16 = 0xFFFF0000u;

/*
// This function extracts the value stored in the left 16 bits of a 32 bit unsigned int.
*/
__device__ unsigned short left_half(unsigned int val){

	
	return (val >> 16u) & MASK_16;

}


/*
// This function extracts the value stored in the right 16 bits of a 32 bit unsigned int.
*/
__device__ unsigned short right_half(unsigned int val){

	return val & MASK_16;

}

/*
// This function concatenates two 16-bit values into one 32-bit field.
*/
__device__ unsigned int merge(unsigned short left, unsigned short right){

	unsigned int result = 0u;
	unsigned int left_int = left;
	unsigned int right_int = right;
	result = ((left_int&MASK_16)<<16u) | (right_int&MASK_16);
	return result;

}


/*
// This function sets the value stored in the right 16 bits of a 32 bit unsigned int.
*/
__device__ void set_right(unsigned int& val, unsigned short right){
	
	unsigned int right_int = right;
	unsigned int local_val = val;
	right_int	&= MASK_16;
	local_val	&= LEFT_16;
	local_val	|= right_int;
	val = local_val;

}


/*
// This function sets the value stored in the left 16 bits of a 32 bit unsigned int.
*/
__device__ void set_left(unsigned int& val, unsigned short left){

	unsigned int left_int = left;
	unsigned int local_val = val;
	left_int	&= MASK_16;
	local_val 	&= MASK_16;
	local_val	|= left_int << 16u;
	val = local_val;

}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions for pairs of 32 bit values packed into 64 bit fields //////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
const unsigned long long int MASK_32 = 0xFFFFFFFFu;
const unsigned long long int LEFT_32 = 0xFFFFFFFF00000000u;

/*
// This function extracts the value stored in the left 32 bits of a 64 bit unsigned long long int.
*/
__device__ unsigned int left_half(unsigned long long int val){

	return (val >> 32u) & MASK_32;

}


/*
// This function extracts the value stored in the right 32 bits of a 64 bit unsigned long long int.
*/
__device__ unsigned int right_half(unsigned long long int val){

	return val & MASK_32;

}

/*
// This function concatenates two 32-bit values into one 64-bit field.
*/
__device__ unsigned long long int merge(unsigned int left, unsigned int right){

	unsigned long long int left_long = left;
	unsigned long long int right_long = right;
	unsigned long long int result = 0u;
	result = ( ( left_long & MASK_32 ) << 32u ) | ( right_long & MASK_32 );
	return result;

}


/*
// This function sets the value stored in the right 32 bits of a 64 bit unsigned long long int.
*/
__device__ void set_right(unsigned long long int& val, unsigned int right){
	
	unsigned long long int right_long = right;
	unsigned long long int local_val = val;
	right_long	&= MASK_32;
	local_val	&= LEFT_32;
	local_val	|= right_long;
	val = local_val;

}


/*
// This function sets the value stored in the left 32 bits of a 64 bit unsigned long long int.
*/
__device__ void set_left(unsigned long long int& val, unsigned long long int left){

	unsigned long long int left_long = left;
	unsigned long long int local_val = val;
	left_long	&= MASK_32;
	local_val 	&= MASK_32;
	local_val	|= left_long << 32u;
	val = local_val;

}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Wrapper functions for queues ///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/*
// This function extracts the head of the given queue 
*/
__device__ ctx_link_adr get_head(ctx_queue queue){

	return left_half(queue.data);

}


/*
// This function extracts the tail of the given queue
*/
__device__ ctx_link_adr get_tail(ctx_queue queue){

	return right_half(queue.data);

}

/*
// This function concatenates two link addresses into a queue
*/
__device__ ctx_queue make_queue(ctx_link_adr head, ctx_link_adr tail){

	ctx_queue result;
	result.data = merge(head,tail);
	return result;

}


/*
// This function sets the head of the queue to the given link address
*/
__device__ void set_head(ctx_queue& queue, ctx_link_adr new_head){
	
	set_left(queue.data,new_head);

}


/*
// This function sets the tail of the queue to the given link address
*/
__device__ void set_tail(ctx_queue& queue, ctx_link_adr new_tail){

	set_right(queue.data,new_tail);

}


__device__ bool is_null(ctx_queue queue){
	return (queue.data == NULL_QUEUE);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// End of packing helper functions ////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////




/*
// Zeros out a link, giving it a thunk count of zero, a null function ID, a null pointer for
// the next link and a depth of zero.
*/
__device__ void empty_link(ctx_link& link, ctx_link_adr next){

	link.depth	= 0;
	link.next	= next;
	link.id		= FUNCTION_ID_COUNT;
	link.count	= 0;

}





/*
// Gives a count of how many threads in the current warp with a lower warp-local id are currently
// active. This is useful for finding an index for a set of operations performed once by an
// unpredictable number of threads within a warp.
*/
__device__ unsigned int warp_inc_scan(){

	unsigned int active = __activemask();
	unsigned int keep = (1 << threadIdx.x) - 1;
	unsigned int scan = __popc(active & keep);
	return scan;

}


/*
// This function returns the number of currently active threads in a warp
*/
__device__ unsigned int active_count(){
	return __popc(__activemask()); 
}


/*
// This returns true only if the current thread is the active thread with the lowest warp-local id.
// This is valuable for electing a "leader" to perform single-threaded work for a warp.
*/
__device__ bool current_leader(){
	return ((__ffs(__activemask())-1) == threadIdx.x);
}


/*
// Returns an index into the partial map of a group based off of a function id and a depth. If
// an invalid depth or function id is used, PART_MAP_NULL is returned.
*/
__device__ unsigned int partial_map_index(unsigned int func_id, unsigned int depth, unsigned int current_level){

	if(func_id >= FUNCTION_ID_COUNT){
		return PART_MAP_NULL;
	}

#if DEF_STACK_MODE == 0
	unsigned int result = func_id;
#else
	unsigned int result = func_id*3;
	if( depth == current_level ){
		result += 1;
	} else if ( depth == (current_level+1) ){
		result += 2;
	} else if ( depth != (current_level-1) ){
		result = PART_MAP_NULL;
	}
#endif

	return result;

}


/*
// Initializes the shared state of a work group, which is stored as a ctx_shared struct. This
// is mainly done by initializing handles to the arena, pool, and stack, setting the current
// level to null, setting the stash iterator to null, and zeroing the stash.
*/
__device__ void init_shared(ctx_shared& shr, ctx_link* arena, ctx_padded_queue* pool, ctx_stack* stack){

	unsigned int active = __activemask();

	__syncwarp(active);

	if(current_leader()){

		int index = threadIdx.x;


		shr.arena = arena;
		shr.pool  = pool;
		shr.stack = stack;

		#if DEF_STACK_MODE == 0
		shr.level = 0;
		#else
		shr.level = NULL_LEVEL;
		#endif

		shr.stash_count = 0;
		shr.link_stash_count = 0;
		shr.keep_running = true;
		shr.busy 	 = false;
		shr.can_make_work= true;
		shr.exec_head    = STASH_SIZE;
		shr.full_head    = STASH_SIZE;
		shr.empty_head   = 0;
		shr.work_iterator= 0;
		shr.scarce_work  = false;

		for(unsigned int i=index; i<STASH_SIZE; i++){
			empty_link(shr.stash[i],i+1);
		}
			
		for(unsigned int i=index; i<FUNCTION_ID_COUNT; i++){
			shr.partial_map[i] = STASH_SIZE;
		}

		shr.SM_thunk_delta = 0;
		
	}

	__syncwarp(active);

}

/*
// Initializes the local state of a thread, which is just the global id of the thread and the
// state used by the thread to generate random numbers for stochastic choices needed to manage
// the runtime state.
*/
__device__ void init_local(ctx_local& loc){

	loc.thread_id	= (blockIdx.x * blockDim.x) + threadIdx.x;
	loc.rand_state = loc.thread_id;

}


/*
// A simple pseudo-random number generator. This algorithm should never be used for cryptography, 
// it is simply used to generate numbers random enough to reduce collisions for atomic
// instructions performed to manage the runtime state.
*/
__device__ unsigned int random_uint(ctx_shared& shr, ctx_local& loc){

	loc.rand_state = (0x10DCDu * loc.rand_state + 1u);
	return loc.rand_state;

}


/*
// Sets the bits in the event_com field of the stack according to the given flag bits.
*/
__device__ void set_flags(ctx_shared& shr, unsigned int flag_bits){

	atomicOr(&shr.stack->event_com,flag_bits);

}


/*
// Returns the current highest level in the stack. Given that this program is highly parallel,
// this number inherently cannot be trusted. By the time the value is fetched, the stack could
// have a different height or the thread that set the height may not have deposited links in the
// corresponding level yet.
*/
__device__ unsigned int highest_level(ctx_shared& shr, ctx_local& loc){

	return left_half(shr.stack->depth_live);

}


/*
// Returns a reference to the frame at the requested level in the stack.
*/
__device__ ctx_frame& get_frame(ctx_shared& shr, unsigned int level){

	return shr.stack->frames[level];

}


/*
// Copies the contents of the source thunk to the destination. Currently, it is quite simple,
// but it can be modified to further minimize memory usage later.
*/
__device__ void copy_ctx_thunk(ctx_thunk& dst, ctx_thunk& src){

	dst = src;

}


/*
// Copies the contents of the source link to the destination. Currently, it is quite simple,
// but it can be modified to further minimize memory usage later.
*/
__device__ void copy_ctx_link(ctx_link& dst, ctx_link& src){

	dst = src;

}



/*
// Joins two queues such that the right queue is now at the end of the left queue.
//
// WARNING: NOT THREAD SAFE. Only use this on queues that have been claimed from the stack
// atomically. If not, one of the queues manipulated with this function will almost certainly
// become malformed at some point. Woe betide those that do not heed this dire message.
*/
__device__ ctx_queue join_queues(ctx_shared& shr, ctx_local& loc, ctx_queue left_queue, ctx_queue right_queue){

	ctx_queue result;

	/*
	// If either input queue is null, we can simply return the other queue.
	*/
	if( is_null(left_queue) ){
		result = right_queue;		
	} else if ( is_null(right_queue) ){
		result = left_queue;
	} else {

		ctx_link_adr left_tail_adr  = get_tail(left_queue);
		ctx_link_adr right_head_adr = get_head(right_queue);
		ctx_link_adr right_tail_adr = get_tail(right_queue);

		/*
		// Find last link in the queue referenced by left_queue.
		*/
		ctx_link& left_tail = shr.arena[left_tail_adr];

		/*
		// Set the index for the tail's successor to the head of the queue referenced by
		// right_queue.
		*/
		left_tail.next = right_head_adr;

		/* Set the right half of the left_queue handle to index the new tail. */
		set_tail(left_queue,right_tail_adr);
		
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
__device__ ctx_link_adr pop_front(ctx_shared& shr, ctx_local& loc, ctx_queue& queue){

	ctx_link_adr result;
	/*
	// Don't try unless the queue is non-null
	*/
	if( is_null(queue) ){
		result = NULL_LINK;
	} else {
		result = get_head(queue);
		ctx_link_adr next = shr.arena[result].next;
		set_head(queue,next);
		if(next == NULL_LINK){
			set_tail(queue,next);
		} else if ( get_tail(queue) == result ){
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
__device__ void push_back(ctx_shared& shr, ctx_local& loc, ctx_queue& queue, ctx_link_adr link){

	
	shr.arena[link].next = NULL_LINK;
	if( is_null(queue) ){
		queue = make_queue(link,link);
	} else {
		ctx_link_adr tail = get_tail(queue);
		shr.arena[tail].next = link;
		set_tail(queue,link);
	}

}




/*
// Attempts to pull a queue from a range of queue slots, trying each slot starting from the given
// starting index onto the end of the range and then looping back from the beginning. If, after
// trying every slot in the range, no non-null queue was obtained, a NULL_QUEUE value is returned.
*/
__device__ ctx_queue pull_queue(ctx_padded_queue* src, unsigned int start_index, unsigned int range_size, unsigned int& src_index){

	ctx_queue result;
	
	
	/*
	// First iterate from the starting index to the end of the queue range, attempting to
	// claim a non-null queue until either there are no more slots to try, or the atomic
	// swap successfuly retrieves something.
	*/
	for(unsigned int i=start_index; i < range_size; i++){
		result.data = atomicExch(&(src[i].queue.data),NULL_QUEUE);
		if( ! is_null(result) ){
			src_index = i;
			return result;
		}
	}

	/*
	// Continue searching from the beginning of the range to just before the beginning of the
	// previous scan.
	*/
	for(unsigned int i=0; i < start_index; i++){
		result.data = atomicExch(&(src[i].queue.data),NULL_QUEUE);
		if( ! is_null(result) ){
			src_index = i;
			return result;
		}
	}

	q_printf("COULD NOT PULL QUEUE\n");
	/*
	// Return NULL_QUEUE if nothing is found
	*/
	result.data = NULL_QUEUE;
	return result;


}




/*
// Repeatedly tries to push a queue to a destination queue slot by atomic exchanges. If a non
// null queue is ever returned by the exchange, it attempts to merge with a subsequent exchange.
// For now, until correctness is checked, this process repeats a limited number of times. In
// production, this will be an infinite loop, as the function should not fail if correctly 
// implemented.
*/
__device__ void push_queue(ctx_shared& shr, ctx_local& loc, ctx_queue& dest, ctx_queue queue){

	#ifdef INF_LOOP_SAFE
	while(true)
	#else
	for(int i=0; i<RETRY_LIMIT; i++)
	#endif
	{
		ctx_queue swap;
		swap.data = atomicExch(&dest.data,queue.data);
		/*
		// If our swap returns a non-null queue, we are still stuck with a queue that
		// needs to be offloaded to the stack. In this case, claim the queue from the 
		// slot just swapped with, merge the two, and attempt again to place the queue
		// back. With this method, swap failures are bounded by the number of pushes to
		// the queue slot, with at most one failure per push_queue call, but no guarantee
		// of which attempt from which call will suffer from an incurred failure.
		*/
		if( ! is_null(swap) ){
			q_printf("Ugh. We got queue (%d,%d) when trying to push a queue\n",get_head(swap),get_tail(swap));
			ctx_queue other_swap;
			other_swap.data = atomicExch(&dest.data,NULL_QUEUE); 
			queue = join_queues(shr,loc,other_swap,swap);
			q_printf("Merged it to form queue (%d,%d)\n",get_head(queue),get_tail(queue));
		} else {
			q_printf("Finally pushed (%d,%d)\n",get_head(queue),get_tail(queue));
			break;
		}
	}


}




/*
// Claims a link from the link stash. If no link exists in the stash, NULL_LINK is returned.
*/
__device__ ctx_link_adr claim_stash_link(ctx_shared& shr, ctx_local& loc){

	ctx_link_adr link = NULL_LINK;
	unsigned int count = shr.link_stash_count;
	if(count > 0){
		link = shr.link_stash[count-1];
		shr.link_stash_count = count - 1;
	}
	q_printf("New link stash count: %d\n",shr.link_stash_count);
	return link;

}



/*
// Inserts an empty slot into the stash. This should only be called if there is enough space in
// the link stash.
*/
__device__ void insert_stash_link(ctx_shared& shr, ctx_local& loc, ctx_link_adr link){

	unsigned int count = shr.link_stash_count;
	shr.link_stash[count] = link;
	shr.link_stash_count = count + 1;
	q_printf("New link stash count: %d\n",shr.link_stash_count);

}




/*
// Claims an empty slot from the stash and returns its index. If no empty slot exists in the stash,
// then STASH_SIZE is returned.
*/
__device__ unsigned int claim_empty_slot(ctx_shared& shr, ctx_local& loc){

	unsigned int slot = shr.empty_head;
	if(slot != STASH_SIZE){
		shr.empty_head = shr.stash[slot].next;
		db_printf("EMPTY: %d << %d\n",slot,shr.empty_head);
	}
	return slot;

}


/*
// Inserts an empty slot into the stash. This should only be called if there is enough space in
// the link stash.
*/
__device__ void insert_empty_slot(ctx_shared& shr, ctx_local& loc, unsigned int slot){

	shr.stash[slot].next = shr.empty_head;
	db_printf("EMPTY: >> %d -> %d\n",slot,shr.empty_head);
	shr.empty_head = slot;

}


/*
// Claims a full slot from the stash and returns its index. If no empty slot exists in the stash,
// then STASH_SIZE is returned.
*/
__device__ unsigned int claim_full_slot(ctx_shared& shr, ctx_local& loc){

	unsigned int slot = shr.full_head;
	if(slot != STASH_SIZE){
		shr.full_head = shr.stash[slot].next;
		db_printf("FULL : %d << %d\n",slot,shr.full_head);
	}
	return slot;

}


/*
// Inserts a full slot into the stash. This should only be called if there is enough space in
// the link stash.
*/
__device__ void insert_full_slot(ctx_shared& shr, ctx_local& loc, unsigned int slot){

	shr.stash[slot].next = shr.full_head;
	db_printf("FULL : >> %d -> %d\n",slot,shr.full_head);
	shr.full_head = slot;

}




/*
// Attempts to fill the link stash to the given threshold with links. This should only ever
// be called in a single-threaded manner.
*/
__device__ void fill_stash_links(ctx_shared& shr, ctx_local& loc, unsigned int threashold){


	unsigned int active = __activemask();
	__syncwarp(active);

	if( current_leader() ){

		for(int try_itr=0; try_itr < RETRY_LIMIT; try_itr++){
	
			if(shr.link_stash_count >= threashold){
				break;
			}
			/*	
			// Attempt to pull a queue from the pool. This should be very unlikely to fail unless
			// almost all links have been exhausted or the pool size is disproportionately small
			// relative to the number of work groups. In the worst case, this should simply not 
			// allocate any links, and the return value shall report this.
			*/
			unsigned int src_index = NULL_LINK;
			unsigned int start = random_uint(shr,loc)%POOL_SIZE;
			ctx_queue queue = pull_queue(shr.pool,start,POOL_SIZE,src_index);
			q_printf("Pulled queue (%d,%d) from pool %d\n",get_head(queue),get_tail(queue),src_index);

			/*
			// Keep popping links from the queue until the full number of links have been added or
			// the queue runs out of links.
			*/
			for(int i=shr.link_stash_count; i < threashold; i++){
				ctx_link_adr link = pop_front(shr,loc,queue);
				if(link != NULL_LINK){
					insert_stash_link(shr,loc,link);
					q_printf("Inserted link %d into link stash\n",link);
				} else {
					break;
				}
			}
			__threadfence();
			push_queue(shr,loc,shr.pool[src_index].queue,queue);
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
__device__ void spill_stash_links(ctx_shared& shr, ctx_local& loc, unsigned int threashold){

	/*
	// Do not even try if no links can be or need to be removed
	*/
	if(threashold >= shr.link_stash_count){
		q_printf("Nothing to spill...\n");
		return;
	}

	/*
	// Find where in the link stash to begin removing links
	*/

	ctx_queue queue;
	queue.data = NULL_QUEUE;

	/*
	// Connect all links into a queue
	*/
	unsigned int spill_count = shr.link_stash_count - threashold;
	for(unsigned int i=0; i < spill_count; i++){
		
		ctx_link_adr link = claim_stash_link(shr,loc);
		q_printf("Claimed link %d from link stash\n",link);
		push_back(shr,loc,queue,link);

	}

	shr.link_stash_count = threashold;


	/*
	// Push out the queue to the pool
	*/
	q_printf("Pushing queue (%d,%d) to pool\n",get_head(queue),get_tail(queue));
	unsigned int dest_idx = random_uint(shr,loc) % POOL_SIZE;
	__threadfence();
	push_queue(shr,loc,shr.pool[dest_idx].queue,queue);
	
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
__device__ void pop_frame_counters(ctx_shared& shr, unsigned int start_level, unsigned int end_level){


	unsigned int depth_dec = 0;
	unsigned int delta;
	unsigned int result;

	ctx_frame& frame = shr.stack->frames[start_level];

	/*
	// Decrement the residents counter for the start level
	*/
	delta = active_count();
	if(current_leader()){
		result = atomicSub(&frame.children_residents,delta);
		if(result == 0u){
			depth_dec += 1;
		}
	}	

	/*
	// Decrement the children counter for the remaining levels
	*/
	for(int d=(start_level-1); d >= end_level; d--){
		ctx_frame& frame = shr.stack->frames[d];
		delta = active_count();
		if(current_leader()){
			result = atomicSub(&frame.children_residents,delta);
			if(result == 0u){
				depth_dec += 1;
			}
		}
	}

	/*
	// Update the stack base once all other counters have been updated.
	*/
	if(current_leader()){
		result = atomicSub(&(shr.stack->depth_live),depth_dec);
		if(result == 0){
			set_flags(shr,COMPLETION_FLAG);
		}
	}

}


/*
// Repetitively tries to merge the given queue of thunks with the queue at the given index in the
// frame at the given level on the stack. This function currently aborts if an error flag is set
// or if too many merge failures occur, however, once its correctness is verified, this function
// will run forever until the merge is successful, as success is essentially guaranteed by
// the nature of the process.
*/
__device__ void push_thunks(ctx_shared& shr, ctx_local& loc, unsigned int level, unsigned int index, ctx_queue queue, int thunk_delta) {


	ctx_link_adr tail  = get_tail(queue);
	ctx_link_adr head = get_head(queue);
	rc_printf("SM %d: push_thunks(level:%d,index:%d,queue:(%d,%d),delta:%d)\n",blockIdx.x,level,index,tail,head,thunk_delta);
	/*
	// Do not bother pushing a null queue if there is no delta to report
	*/
	if( ( ! is_null(queue) ) || (thunk_delta != 0) ){

		/*
		// Change the resident counter of the destination frame by the number of thunks
		// that have been added to or removed from the given queue
		*/
		ctx_frame &dest = get_frame(shr,level);		
		unsigned int old_count;
		unsigned int new_count;
		if(thunk_delta >= 0) {
			old_count = atomicAdd(&dest.children_residents,(unsigned int) thunk_delta);
			new_count = old_count + (unsigned int) thunk_delta;
			if(old_count > new_count){
				rc_printf("\n\nOVERFLOW\n\n");
			}
		} else {
			unsigned int neg_delta = - thunk_delta;
			old_count = atomicSub(&dest.children_residents,neg_delta);
			new_count = old_count - neg_delta; 
			if(old_count < new_count){
				rc_printf("\n\nUNDERFLOW\n\n");
			}
		}


		shr.SM_thunk_delta += thunk_delta;
		rc_printf("SM %d-%d: Old count: %d, New count: %d\n",blockIdx.x,threadIdx.x,old_count,new_count);

		
		//rc_printf("SM %d-%d: frame zero resident count is: %d\n",blockIdx.x,threadIdx.x,shr.stack->frames[0].children_residents);
		/*
		// If the addition caused a frame to change from empty to non-empty or vice-versa,
		// make an appropriate incrementation or decrementation at the stack base.
		*/
		if( (old_count == 0) && (new_count != 0) ){
			atomicAdd(&(shr.stack->depth_live),0x00010000u);
		} else if( (old_count != 0) && (new_count == 0) ){
			atomicSub(&(shr.stack->depth_live),0x00010000u);
		} else {
			rc_printf("SM %d: No change!\n",blockIdx.x);
		}

		__threadfence();	
		/*
		// Finally, push the queues
		*/
		push_queue(shr,loc,dest.queues[index].queue,queue);
		rc_printf("SM %d: Pushed queue (%d,%d) to stack at index %d\n",blockIdx.x,get_head(queue),get_tail(queue),index);
	
			
		//rc_printf("SM %d-%d: After queue pushed to stack, frame zero resident count is: %d\n",blockIdx.x,threadIdx.x,shr.stack->frames[0].children_residents);

	}
	rc_printf("(%d) the delta: %d\n",blockIdx.x,thunk_delta);

}



/*
// Attempts to pull a queue of thunks from the frame in the stack of the given level, starting the 
// pull attempt at the given index in the frame. If no queue could be pulled after attempting a 
// pull at each queue in the given frame, a NULL_QUEUE value is returned.
*/
__device__ ctx_queue pull_thunks(ctx_shared& shr, ctx_local& loc, unsigned int level, unsigned int& source_index) {


	rc_printf("SM %d: pull_thunks(level:%d)\n",blockIdx.x,level);
	unsigned int src_idx = random_uint(shr,loc) % QUEUES_PER_FRAME;

	ctx_frame &src = get_frame(shr,level);		
	
	ctx_queue queue = pull_queue(src.queues,src_idx,QUEUES_PER_FRAME,source_index);
	if( ! is_null(queue) ){
		q_printf("SM %d: Pulled queue (%d,%d) from stack at index %d\n",blockIdx.x,get_tail(queue),get_head(queue), src_idx);
	} else {
		q_printf("SM %d: Failed to pull queue from stack starting at index %d\n",blockIdx.x, src_idx);
	}
	return queue;

}


/*
// Attempts to pull a queue from any frame in the stack, starting from the highest and working
// its way down. If no queue could be pulled, a NULL_QUEUE value is returned.
*/
__device__ ctx_queue pull_thunks_any_level(ctx_shared& shr, ctx_local& loc, unsigned int& level, unsigned int& source_index){


	ctx_queue result;
	result.data = NULL_QUEUE;
	unsigned int start_level = highest_level(shr,loc);
	for(int level_itr = start_level; level_itr>=0; level_itr--){
		q_printf("Pulling thunks at level %d for pull_thunks_any_level\n",level_itr);
		result = pull_thunks(shr,loc,level_itr,source_index);
		if( ! is_null(result) ){
			level = level_itr;
			return result;
		}
	}
	result.data = NULL_QUEUE;
	return result;

}





/*
// Adds the contents of the stash slot at the given index to a link and returns the index of the 
// link in the arena. This should only ever be called if there is both a link available to store
// the data and if the index is pointing at a non-empty slot. This also should only ever be
// called in a single-threaded context.
*/
__device__ ctx_link_adr produce_link(ctx_shared& shr, ctx_local& loc, unsigned int slot_index ){


	__shared__ ctx_link_adr result;

	unsigned int active = __activemask();


	//__syncwarp(active);
	
	//if(current_leader()){
		unsigned int link_index = claim_stash_link(shr,loc);
		db_printf("Claimed link %d from stash\n",link_index);
		ctx_link& the_link = shr.arena[link_index];
		//shr.SM_thunk_delta += shr.stash[slot_index].count;
		shr.stash[slot_index].next = NULL_LINK;
		copy_ctx_link(the_link, shr.stash[slot_index]);
		db_printf("Link has count %d and next %d in global memory",the_link.count, the_link.next);
		result = link_index;
		shr.stash_count -= 1;
	//}

	
	//__syncwarp(active);
	return result;

}








/*
// Removes all thunks in the stash that do not correspond to the given level, or to the levels
// immediately above or below (level+1) and (level-1).
*/
__device__ void relevel_stash(ctx_shared& shr, ctx_local& loc, unsigned int level){

#if DEF_STACK_MODE != 0
	unsigned int active =__activemask();
	__syncwarp(active);
	
	__shared__ unsigned int queues[3];
	__shared__ unsigned int counts[3];
	
	__shared__ unsigned int range;
	__shared__ unsigned int start;
	__shared__ unsigned int dump_count;
	

	/*
	// Currently implemented in a single-threaded manner per work group to simplify the initial
	// correctness checking process. This can later be changed to take advantage of in-group
	// parallelism.
	*/
	if(current_leader()){

		/*
		// Determine what range of depths need to be dumped to the stack.
		*/
		unsigned int old_low, old_high, new_low, new_high;
		old_low  = (shr.level == 0) ? 0 : shr.level-1;
		new_low  = (level == 0)     ? 0 : level-1;
		old_high = (shr.level == MAX_LEVEL) ? MAX_LEVEL : level+1;
		new_high = (level == MAX_LEVEL)     ? MAX_LEVEL : level+1;

		if( (old_high < new_low) || (new_high < old_low) ) {
			start = old_low;
			range = old_high-old_low;
		} else if (new_low >= old_low) {
			start = old_low;
			range = new_low - old_low;
		} else {
			start = new_high+1;
			range = old_high-new_high-1;
		}


		/*
		// Zero out counters and accumulating queues.
		*/
		dump_count = 0;

		if(range > 0) {
			for(unsigned int i=0; i < range; i++){
				counts[i] = 0;
				queues[i] = NULL_QUEUE;
			}

			/*
			// Find how many links we need to dump unwanted thunks to the stack.
			*/
			for(unsigned int i=0; i < STASH_SIZE; i++){
				unsigned int depth = shr.stash[i].depth;
				if( (depth >= start) && (depth <= (start + range) ) ){
					dump_count++;
				}
			}
		}

		/*
		// If at least one link needs to be dumped, we iterate through again to accumulate
		// the appropriate links into queues for dumping
		*/
		if(dump_count > 0){

			/*
			// Make sure we have enough links in the link stash to build a properly
			// sized queue for dumping.
			*/
			if(dump_count > shr.link_stash_count){
				unsigned int alloc_count = dump_count-shr.link_stash_count;
				for(unsigned int i = 0; i < RETRY_LIMIT; i++){
					alloc_links(shr,loc,alloc_count);
				}
			}

			/*
			// Copy each link to be dumped out of work group memory into the global
			// memory occupied by our allocated links, zeroing out each link in the
			// stash once it has been copied out so that work does not get duplicated.
			*/
			for(unsigned int i=0; i < STASH_SIZE; i++){

				unsigned int depth = shr.stash[i].depth;
				
				if( (depth < start) || (depth >= (start+range) ) ){
					continue;
				}

				shr.stash[i].next = NULL_LINK;
				unsigned int link_idx = shr.link_stash[shr.link_stash_count];
				shr.link_stash_count --;
				copy_ctx_link(shr.arena[link_idx],shr.stash[i]);

				unsigned int level_idx = depth-start;
				counts[level_idx] += shr.stash[i].size;
				queues[level_idx] = join_queues(shr,loc,queues[level_idx],link_idx);

				shr.stash[i].size = 0;
			}

			/*
			// Dump the accumulated links to the stack
			*/
			for(unsigned int i=0; i < range; i++){
				unsigned int push_idx = random_uint(shr,loc)%QUEUES_PER_FRAME;
				push_thunks(shr,loc,start+i,push_idx,queues[i],counts[i] );		
			}
		}
	
	}

	__syncwarp(active);

#endif

}





/*
// Dumps all full links not corresponding to the current execution level. Furthermore, should the
// remaining links still put the stash over the given threshold occupancy, links will be further
// removed in the order: full links at the current level, partial links not at the current level,
// partial links at the current level. 
*/
__device__ void spill_stash(ctx_shared& shr, ctx_local& loc, unsigned int threashold){

	unsigned int active =__activemask();
	__syncwarp(active);


	
#if DEF_STACK_MODE == 0


	if(current_leader()){
		db_printf("Current stash_count: %d\n",shr.stash_count);
	}
	
	
	if(current_leader() && (shr.stash_count > threashold)){

		unsigned int spill_count = shr.stash_count - threashold;
		int delta = 0;
		fill_stash_links(shr,loc,spill_count);
		
		ctx_queue queue;
		queue.data = NULL_QUEUE;
		unsigned int partial_iter = 0;
		bool has_full_slots = true;
		for(unsigned int i=0; i < spill_count; i++){
			unsigned int slot = STASH_SIZE;
			if(has_full_slots){
				slot = claim_full_slot(shr,loc);
				if(slot == STASH_SIZE){
					has_full_slots = false;
				}
			}
			if(! has_full_slots){
				for(;partial_iter < FUNCTION_ID_COUNT; partial_iter++){
					db_printf("%d",partial_iter);
					if(shr.partial_map[partial_iter] != STASH_SIZE){
						slot = shr.partial_map[partial_iter];
						partial_iter++;
						break;
					}
				}
			}
			if(slot == STASH_SIZE){
				break;
			}
			
			delta += shr.stash[slot].count;
			db_printf("Slot for production (%d) has %d thunks\n",slot,shr.stash[slot].count);
			ctx_link_adr link = produce_link(shr,loc,slot);
			push_back(shr,loc,queue,link);
			insert_empty_slot(shr,loc,slot);
			if(shr.stash_count <= threashold){
				break;
			}
		}
	
		
		unsigned int push_index = random_uint(shr,loc)%QUEUES_PER_FRAME;
		rc_printf("Pushing thunks for spilling\n");
		push_thunks(shr,loc,0,push_index,queue,delta);
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
	if(current_leader()){

		/*
		// Zero the counters and null the queues
		*/
		for(unsigned int i=0; i < 3; i++){
			queues[i] = NULL_QUEUE;
			counts[i] = 0;
		}

		for(unsigned int i=0; i < 4; i++){
			bucket[i] = 0;
		}


		/*
		// Count up each type of link
		*/
		for(unsigned int i=0; i < STASH_SIZE; i++){
			unsigned int depth = shr.stash[i].depth;
			unsigned int size = shr.stash[i].size;
			unsigned int idx = (depth != level) ? 0 : 1;
			idx += (size >= WARP_COUNT) ? 0 : 2;
			bucket[idx] += 1;
		} 

		/*
		// Determine how much of which type of link needs to be dumped
		*/
		unsigned int dump_total = bucket[0];
		unsigned int dump_count = (shr.stash_count > threshold) ? shr.stash_count - threshold : 0;
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
		for(unsigned int i=0; i < shr.stash_count; i++){
			unsigned int depth = shr.stash[i].depth;
			unsigned int size  = shr.stash[i].size;
			unsigned int bucket_idx = (depth != level) ? 0 : 1;
			bucket_idx += (size >= WARP_COUNT) ? 0 : 2;
			if(bucket[bucket_idx] == 0){
				continue;
			}
			ctx_link_adr link = shr.link_stash[shr.link_stash_count];
			shr.link_stash_count -= 1;

			copy_link(shr.arena[link], shr.stash[i]);

			unsigned int level_index = level+1-depth;
			counts[level_index] += size;
			push_back(shr,loc,queues[level_index],link);

			shr.stash[i].size = 0;
		}
	}

#endif
	
	__syncwarp(active);
	

 
}






/*
// Queues the input thunk into a corresponding local link according to the given function id and
// at a level corresponding to the current level of he thunks being evaluated plus the value of
// depth_delta. This scheme ensures that the function being called and the depth of the thunks
// being created for those calls are consistent across the warp.
*/
__device__ void async_call(ctx_shared& shr, ctx_local& loc, unsigned int func_id, int depth_delta, ctx_thunk thunk){

	unsigned int active = __activemask();

	/*
	// Calculate how many thunks are being queued as well as the assigned index of the 
	// current thread's thunk in the write to the stash.
	*/
	unsigned int index = warp_inc_scan();
	unsigned int delta = active_count();


	/*
	// Make room to queue incoming thunks, if there isn't enough room already.
	*/
	#if 0
	if(shr.link_stash_count < 1){
		fill_stash_links(shr,loc,1);
	}

	if(shr.stash_count >= (STASH_SIZE-1)){
		spill_stash(shr,loc, STASH_SIZE-2);
	}
	#else
	unsigned int depth = (unsigned int) (shr.level + depth_delta);
	unsigned int left_jump = partial_map_index(func_id,depth,shr.level);
	unsigned int space = 0;
	if( left_jump != PART_MAP_NULL ){
		unsigned int left_idx = shr.partial_map[left_jump];	
		if( left_idx != STASH_SIZE ){
			space = WG_SIZE - shr.stash[left_idx].count;
		}
	}
	if( (shr.stash_count >= (STASH_SIZE-1)) && (space < delta) ){
		if(shr.link_stash_count < 1){
			fill_stash_links(shr,loc,1);
		}
		spill_stash(shr,loc, STASH_SIZE-2);
	}
	#endif



	__shared__ unsigned int left, left_start, right;


	/*
	// Locate the destination links in the stash that the thunks will be written to. For now,
	// like many other parts of the code, this will be single-threaded within the work group
	// to make validation easier but will be optimized for group-level parallelism later.
	*/
	if( current_leader() ){

		db_printf("Queueing %d thunks of type %d\n",delta,func_id);
		/*
		// Null out the right index. This index should not be used unless the number of
		// thunks queued spills over beyond the first link being written to (the left one)
		*/
		right = STASH_SIZE;

		/*
		// Find the index of the partial link in the stash corresponding to the id and
		// depth of the calls being queued (if it exists).
		*/
		unsigned int depth = (unsigned int) (shr.level + depth_delta);
		unsigned int left_jump = partial_map_index(func_id,depth,shr.level);
		
		/*
		// If there is a partially filled link to be filled, assign that to the left index
		*/
		if(left_jump != PART_MAP_NULL){
			//db_printf("A\n");
			left = shr.partial_map[left_jump];
		}

		unsigned int left_count;
		if(left == STASH_SIZE){
			//db_printf("B\n");
			left = claim_empty_slot(shr,loc);
			shr.stash_count += 1;
			db_printf("Updated stash count: %d\n",shr.stash_count);
			shr.stash[left].id    = func_id;
			shr.partial_map[left_jump] = left;
			left_count = 0;
		} else {
			left_count = shr.stash[left].count;
		}

		if ( (left_count + delta) > WG_SIZE ){
			//db_printf("C\n");
			right = claim_empty_slot(shr,loc);
			shr.stash_count += 1;
			db_printf("Updated stash count: %d\n",shr.stash_count);
			shr.stash[right].count = left_count+delta - WG_SIZE;
			shr.stash[right].id    = func_id;
			insert_full_slot(shr,loc,left);
			shr.partial_map[left_jump] = right;
			shr.stash[left].count = WG_SIZE;
		} else if ( (left_count + delta) == WG_SIZE ){
			//db_printf("D\n");
			shr.partial_map[left_jump] = STASH_SIZE;
			insert_full_slot(shr,loc,left);
			shr.stash[left].count = WG_SIZE;
		} else {
			shr.stash[left].count = left_count + delta;
		}

		left_start = left_count;


	}
	
	/*
	// Write the thunk into the appropriate part of the stash, writing into the left link 
	// when possible and spilling over into the right link when necessary.
	*/
	__syncwarp(active);
	if( (left_start + index) >= WG_SIZE ){
		//db_printf("Overflow: id: %d, left: %d, left_start: %d, index: %d\n",threadIdx.x,left,left_start,index);
		copy_ctx_thunk(shr.stash[right].data.data[left_start+index-WG_SIZE],thunk);
	} else {
		//db_printf("Non-overflow: id: %d, left: %d, left_start: %d, index: %d\n",threadIdx.x,left,left_start,index);
		copy_ctx_thunk(shr.stash[left].data.data[left_start+index],thunk);
	}
	__syncwarp(active);

	

}




//#define PARACON

/*
// Adds the contents of the link at the given index to the stash and adds the given link to link
// stash. Once complete, it returns the number of thunks added to the stash by the operation.
// This should only ever be called if there is enough space to store the extra work and link.
*/
__device__ unsigned int consume_link(ctx_shared& shr, ctx_local& loc, ctx_link_adr link_index ){


	#ifdef PARACON
	__shared__ ctx_link_adr the_index;
	__shared__ unsigned int add_count;
	__shared__ unsigned int func_id;

	unsigned int active = __activemask();


	__syncwarp(active);
	
	if(current_leader()){

		the_index = link_index;
		add_count = shr.arena[link_index].count;
		func_id   = shr.arena[link_index].id;

	}

	
	__syncwarp(active);
	
	if(threadIdx.x < add_count){
		ctx_thunk thunk = shr.arena[the_index].data.data[threadIdx.x];
		async_call(shr,loc,func_id,0,thunk);
	}


	__syncwarp(active);


	if(current_leader()){
		insert_stash_link(shr,loc,link_index);
	}

	return add_count;


	#else 

	ctx_link_adr the_index;
	unsigned int add_count;
	unsigned int func_id;

	unsigned int active = __activemask();
	unsigned int acount = active_count();

	

	the_index = link_index;
	add_count = shr.arena[link_index].count;
	func_id   = shr.arena[link_index].id;

	//shr.SM_thunk_delta -= add_count;
	
	db_printf("active count: %d, add count: %d\n",acount,add_count);

	
	db_printf("\n\nprior stash count: %d\n\n\n",shr.stash_count);
	//*
	for(unsigned int i=0; i< add_count; i++){
		ctx_thunk thunk = shr.arena[the_index].data.data[i];
		async_call(shr,loc,func_id,0,thunk);
	}
	// */
	//ctx_thunk thunk = shr.arena[the_index].data.data[0];
	//async_call(shr,loc,func_id,0,thunk);

	db_printf("\n\nafter stash count: %d\n\n\n",shr.stash_count);


	insert_stash_link(shr,loc,link_index);

	return add_count;



	#endif


}






/*
// Tries to transfer links from the stack into the stash of the work group until the stash
// is filled to the given threashold. If a halting condition is reached, this function will set
// the keep_running value in the shared context to false.
*/
__device__ void fill_stash(ctx_shared& shr, ctx_local& loc, unsigned int threashold){

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
	if(current_leader()){

		//db_printf("Filling stash...\n");

		unsigned int taken = 0;
		
		threashold = (threashold > STASH_SIZE) ? STASH_SIZE : threashold;
		unsigned int gather_count = (threashold < shr.stash_count) ? 0  : threashold - shr.stash_count;
		if( (STASH_SIZE - shr.link_stash_count) < gather_count){
			unsigned int spill_thresh = STASH_SIZE - gather_count;
			spill_stash_links(shr,loc,spill_thresh);
		}
		

		#ifdef PARACON
		unsigned int loc_link_count = 0;
		#endif

		#ifdef RACE_COND_PRINT
		unsigned int p_depth_live = shr.stack->depth_live;
		rc_printf("SM %d: depth_live is (%d,%d)\n",blockIdx.x,left_half(p_depth_live),right_half(p_depth_live));
		#endif

		for(unsigned int i = 0; i < RETRY_LIMIT; i++){

			/* If the stack is empty or a flag is set, return false */
			unsigned int depth_live = shr.stack->depth_live;
			if( (depth_live == 0u) || ( shr.stack->event_com != 0u) ){
				shr.keep_running = false;
				break;
			}


			unsigned int src_index;
			ctx_queue queue;

			#if DEF_STACK_MODE == 0
		
			db_printf("STACK MODE ZERO\n");	
			q_printf("%dth try pulling thunks for fill\n",i+1);
			queue = pull_thunks(shr,loc,shr.level,src_index);

			#else
			/*
			// Determine whether or not to pull from the current level in the stack
			*/
			unsigned int depth = left_half(depth_live);
			bool pull_any = (depth < shr.level);
			ctx_frame &current_frame = get_frame(shr,depth);
			if(!pull_any){
				pull_any = (right_half(current_frame.children_residents) == 0);
			}


			/*
			// Retrieve a queue from the stack.
			*/

			if(pull_any){
				unsigned int new_level;
				queue = pull_thunks_any_level(shr,loc,new_level,src_index);
				relevel_stash(shr,loc,new_level);
			} else {
				queue = pull_thunks(shr,loc,shr.level,src_index);
			}
			#endif

			#ifdef PARACON
			db_printf("About to pop thunks\n");
			while(	( ! is_null(queue) ) 
			     && (loc_link_count < gather_count)
			     && (shr.link_stash_count < STASH_SIZE) 
			){
				ctx_link_adr link = pop_front(shr,loc,queue);					
				if(link != NULL_LINK){
					db_printf("Popping front %d\n",link);
					links[loc_link_count] = link;
					taken += shr.arena[link].count;
					loc_link_count++;
				} else {
					break;
				}
			}
			#else
			db_printf("About to pop thunks\n");
			while(	( ! is_null(queue) ) 
			     && (shr.stash_count < threashold)
			     && (shr.link_stash_count < STASH_SIZE) 
			){
				ctx_link_adr link = pop_front(shr,loc,queue);					
				if(link != NULL_LINK){
					q_printf("Popping front %d\n",link);
					taken += consume_link(shr,loc,link);
				} else {
					break;
				}
			}
			#endif
	
			db_printf("Popped thunks\n");
			if(taken != 0){
				if(!shr.busy){
					atomicAdd(&(shr.stack->depth_live),1);
					shr.busy = true;
					rc_printf("SM %d: Incremented depth value\n",blockIdx.x);
				}
				rc_printf("Pushing thunks for filling\n");	
				push_thunks(shr,loc,shr.level,src_index,queue,-taken);
				break;
			}
	
			#ifndef PARACON
			if( shr.stash_count >= threashold ){
				break;
			}
			#endif

		}
	




		if(shr.busy && (shr.stash_count == 0)){
			atomicSub(&(shr.stack->depth_live),1);
			rc_printf("SM %d: Decremented depth value\n",blockIdx.x);
			shr.busy = false;
		}


		#ifdef PARACON
		link_count = loc_link_count;
		#endif

		
	}

	__syncwarp(active);



	#ifdef PARACON

	for(int i=0; i<link_count;i++){
		consume_link(shr,loc,links[i]);
	}


	__syncwarp(active);
	#endif



}




__device__ void clear_exec_head(ctx_shared& shr, ctx_local& loc){

	
	if( current_leader() && (shr.exec_head != STASH_SIZE) ){
		insert_empty_slot(shr,loc,shr.exec_head);
		shr.exec_head = STASH_SIZE;
	}
	__syncwarp();

}




/*
// Selects the next link in the stash. This selection process could become more sophisticated
// in later version to account for the average branching factor of each async function. For now,
// it selects the fullest slot of the current level if it can. If no slots with thunks for the
// current level exist in the stash, the function returns false.
*/
__device__ bool advance_stash_iter(ctx_shared& shr, ctx_local& loc){

	__shared__ bool result;
	unsigned int active =__activemask();
	__syncwarp(active);
	

	if(current_leader()){

		if(shr.full_head != STASH_SIZE){
			shr.exec_head = claim_full_slot(shr,loc);
			shr.stash_count -= 1;
			result = true;
			//db_printf("Found full slot.\n");
		} else {
			//db_printf("Looking for partial slot...\n");
			unsigned int best_id   = PART_MAP_NULL;
			unsigned int best_slot = STASH_SIZE;
			unsigned int best_count = 0;
			for(int i=0; i < FUNCTION_ID_COUNT; i++){
				unsigned int slot = shr.partial_map[i];
				
				if( (slot != STASH_SIZE) && (shr.stash[slot].count > best_count)){
					best_id = i;
					best_slot = slot;
					best_count = shr.stash[slot].count;
				}
				
			}

			result = (best_slot != STASH_SIZE);
			if(result){
				//db_printf("Found partial slot.\n");
				shr.exec_head = best_slot;
				shr.partial_map[best_id] = STASH_SIZE;
				shr.stash_count -=1;
			}
		}

	}

	__syncwarp(active);
	return result;

}





/*
// This function, as well as the functions directly called within this switch statement will
// eventually need to be created by the programmer to incorperate their code into the system.
// This function, specifically, could be auto-generated in response to a programmer's async
// functions as the project matures.
*/
__device__ void do_async(ctx_shared& shr,ctx_local& loc, unsigned int func_id,ctx_thunk& frame);

/*
// This function, as well as the functions directly called within this switch statement will
// eventually need to be created by the programmer to incorperate their code into the system.
// This function, specifically, could be auto-generated in response to a programmer's async
// functions as the project matures.
*/
__device__ void make_work(ctx_shared& shr,ctx_local& loc);








/*
// Tries to perform up to one work group worth of work by selecting a link from shared memory (or,
// if necessary, fetching a link from global memory), and running the function on the data within
// the link, as directed by the function id the link is labeled with. This function returns false
// if a halting condition has been reached (either due to lack of work or an event) and true
// otherwise.
*/
__device__ void exec_cycle(ctx_shared& shr, ctx_local& loc){


	clear_exec_head(shr,loc);

	/*
	// Advance the stash iterator to the next chunk of work that needs to be done.
	*/
	//*
	if( shr.can_make_work && (shr.full_head == STASH_SIZE) ){
		make_work(shr,loc);
		if( current_leader() && (! shr.busy ) && ( shr.stash_count != 0 ) ){
			atomicAdd(&(shr.stack->depth_live),1);
			shr.busy = true;
		}
	}


	#if 1
	if(shr.full_head == STASH_SIZE){
		fill_stash(shr,loc,STASH_SIZE-2);
	}
	#else
	if(shr.full_head == STASH_SIZE){
		if( !shr.scarce_work ){
			fill_stash(shr,loc,STASH_SIZE-2);
			if( current_leader() && (shr.full_head == STASH_SIZE) ){
				shr.scarce_work = true;
			}
		}
	} else {
		if( current_leader() ){
			shr.scarce_work = false;
		}
	}
	#endif
	// */

	if( !advance_stash_iter(shr,loc) ){
		/*
		// No more work exists in the stash, so try to fetch it from the stack.
		*/
		fill_stash(shr,loc,STASH_SIZE-2);

		if( shr.keep_running && !advance_stash_iter(shr,loc) ){
			/*
			// REALLY BAD: The fill_stash function successfully, however 
			// the stash still has no work to perform. In this situation,
			// we set an error flag and halt.
			*/
			/*
			if(current_leader()){
				db_printf("\nBad stuff afoot!\n\n");
			}
			set_flags(shr,STASH_FAIL_FLAG);
			shr.keep_running = false;
			*/
		}
	}

	
	unsigned int active = __activemask();
	__syncwarp(active);


	if( (shr.exec_head != STASH_SIZE) && shr.keep_running ){
		/* 
		// Find which function the current link corresponds to.
		*/	
		unsigned int func_id     = shr.stash[shr.exec_head].id;
		unsigned int thunk_count = shr.stash[shr.exec_head].count;
		
		/*
		// Only execute if there is a thunk in the current link corresponding to the thread that
		// is being executed.
		*/
		if(current_leader()){
			q_printf("Executing slot %d, which is %d thunks of type %d\n",shr.exec_head,thunk_count,func_id);
		}
		if( threadIdx.x < thunk_count ){
			//db_printf("Executing...\n");
			ctx_thunk& thunk = shr.stash[shr.exec_head].data.data[threadIdx.x];
			do_async(shr,loc,func_id,thunk);
		}
	}

	__syncwarp(active);


}



__device__ void cleanup_runtime(ctx_shared& shr, ctx_local& loc){

	unsigned int active = __activemask();
	__syncwarp(active);
	if(current_leader()){
		q_printf("CLEANING UP\n");
		spill_stash(shr,loc,0);
		spill_stash_links(shr,loc,0);	
		if(shr.busy){
			atomicSub(&(shr.stack->depth_live),1);
		}

	}




}



struct runtime_context{
	ctx_link*  		arena;
	ctx_padded_queue*	pool;
	ctx_stack*		stack;
};

/*
//
// This must be run once on the resources used for execution, prior to execution. Given that this
// essentially wipes all data from these resources and zeros all values, it is not advised that
// this function be used at any other time, except to setup for a re-start or to clear out after
// calling the pull_runtime to prevent thunk duplication.
//
*/
__global__ void init_runtime(runtime_context runtime){

	/* Initialize per-warp resources */
	__shared__ ctx_shared shr;
	init_shared(shr,runtime.arena,runtime.pool,runtime.stack);
	
	/* Initialize per-thread resources */
	ctx_local loc;
	init_local(loc);	


	const unsigned int threads_per_frame = QUEUES_PER_FRAME + 1;

	const unsigned int total_stack_work = STACK_SIZE * threads_per_frame;

	/*
	// If the currently executing thread has global thread index 0, wipe the data in the base
	// of the stack.
	*/
	if(loc.thread_id == 0){
		shr.stack->event_com  = 0;
		shr.stack->depth_live = 0;
	}
	/*
	// Blank out the frames in the stack. Setting queues to NULL_QUEUE, and zeroing the counts
	// for resident thunks and child thunks of each frame.
	*/
	for(unsigned int base_index = 0u; base_index < total_stack_work; base_index+=WORKER_COUNT ){
		
		unsigned int index = base_index + loc.thread_id;
		unsigned int target_level = index / threads_per_frame;
		unsigned int frame_index  = index % threads_per_frame;
		if( frame_index == QUEUES_PER_FRAME ){
			shr.stack->frames[target_level].children_residents = 0u;
		} else {
			shr.stack->frames[target_level].queues[threadIdx.x].queue.data = NULL_QUEUE;
		}

	}


	/*
	// Initialize the arena, connecting the contained links into roughly equally sized lists,
	// zeroing the thunk counter in the links and marking the function ID with an invalid
	// value to make use-before-initialization more obvious during system validation.
	*/
	unsigned int bump = ((ARENA_SIZE%POOL_SIZE) != 0) ? 1 : 0;
	unsigned int arena_init_stride = ARENA_SIZE/POOL_SIZE + bump;
	for(unsigned int base_index = 0u; base_index < ARENA_SIZE; base_index+=WORKER_COUNT ){
		
		unsigned int index = base_index + loc.thread_id;
		if(index >= ARENA_SIZE){
			break;
		}
		unsigned int next = index + 1;
		if( ( (next % arena_init_stride) == 0 ) || (next >= ARENA_SIZE) ){
			next = NULL_LINK;
		}
		empty_link(shr.arena[index],next);

	}


	/*
	// Initialize the pool, giving each queue slot one of the previously created linked lists.
	*/
	for(unsigned int base_index = 0u; base_index < POOL_SIZE; base_index+=WORKER_COUNT ){	
		
		unsigned int index = base_index + loc.thread_id;
		if(index >= POOL_SIZE){
			break;
		}
		unsigned int head = arena_init_stride * index;
		unsigned int tail = arena_init_stride * (index + 1) - 1;
		tail = (tail >= ARENA_SIZE) ? ARENA_SIZE - 1 : tail;
		shr.pool[index].queue = make_queue(head,tail);

	}

}


/*
// Unpacks all thunk data from the call buffer into the runtime stack. This
// could be useful for backing up program states for debugging or to re-start processing from
// a previous state.
//
*/
__global__ void push_runtime(runtime_context runtime, ctx_link* call_buffer, unsigned int link_count){


	/* Initialize per-warp resources */
	__shared__ ctx_shared shr;
	init_shared(shr,runtime.arena,runtime.pool,runtime.stack);
	
	/* Initialize per-thread resources */
	ctx_local loc;
	init_local(loc);	


	for(int link_index=blockIdx.x; link_index < link_count; link_index+= WG_COUNT){
		ctx_link& the_link = call_buffer[link_index];
		unsigned int count   = the_link.count;
		unsigned int func_id = the_link.id;
		if(threadIdx.x < count){
			db_printf("\nasync_call(id:%d,depth: 0)\n\n",func_id);
			async_call(shr,loc,func_id,0,the_link.data.data[threadIdx.x]);
		}

	}

	cleanup_runtime(shr,loc);

}




void checkError(){

	cudaError_t status = cudaGetLastError();

	if(status != cudaSuccess){
		const char* err_str = cudaGetErrorString(status);
		printf("ERROR: \"%s\"\n",err_str);
	}

}





/*
// Places a single function call into the runtime.
*/
void remote_call(runtime_context runtime, unsigned int func_id, ctx_thunk thunk){
	
	ctx_link* call_buffer;
	cudaMalloc( (void**) &call_buffer, sizeof(ctx_link) );

	ctx_link host_link;
	host_link.count		= 1;
	host_link.id    	= func_id;
	host_link.next    	= NULL_LINK;
	host_link.depth    	= 0;
	host_link.meta_data.data= 0;
	host_link.data.data[0]	= thunk;

	cudaMemcpy(call_buffer,&host_link,sizeof(ctx_link),cudaMemcpyHostToDevice);

	
	push_runtime<<<WG_COUNT,WG_SIZE>>>(runtime,call_buffer,1);

	checkError();
	
	cudaFree(call_buffer);

} 





/*
// Packs all thunk data from the runtime stack into the communication buffer (comm_buffer). This
// could be useful for backing up program states for debugging or to re-start processing from
// a previous state.
//
// For now, this will not be implemented, as it isn't particularly useful until the system's
// correctness has been verified.
//
*/
__global__ void pull_runtime(runtime_context runtime){

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
__global__ void exec_runtime(runtime_context runtime, unsigned int cycle_count){

	

	/* Initialize per-warp resources */
	__shared__ ctx_shared shr;
	init_shared(shr,runtime.arena,runtime.pool,runtime.stack);
	
	/* Initialize per-thread resources */
	ctx_local loc;
	init_local(loc);	


	if(current_leader()){
		rc_printf("\n\n\nInitial frame zero resident count is: %d\n\n\n",runtime.stack->frames[0].children_residents);
	}	

	/* The execution loop. */
	#ifdef RACE_COND_PRINT
	unsigned int cycle_break = cycle_count;
	#endif
	for(unsigned int cycle=0u; cycle<cycle_count; cycle++){
		/* Early halting handled with a break. */
		exec_cycle(shr,loc);
		if(!shr.keep_running){
			#ifdef RACE_COND_PRINT
			cycle_break = cycle+1;
			#endif
			break;
		}		
	}
	/*
	// Ensure that nothing which should persist between dispatches is lost in the
	// shared or private memory of the halting program.
	*/
	cleanup_runtime(shr,loc);

		
	if(current_leader()){
		rc_printf("SM %d finished after %d cycles with thunk delta %d\n",blockIdx.x,cycle_break,shr.SM_thunk_delta);
	}

}












bool queue_count(ctx_link* host_arena, ctx_queue queue, ctx_link_adr& result){

	ctx_link_adr head = (queue.data >> LINK_BITS) & LINK_MASK;
	ctx_link_adr tail = queue.data & LINK_MASK;
	ctx_link_adr last = NULL_LINK;
	ctx_link_adr count = 0;	
	
	if( head == NULL_LINK ){
		if( tail == NULL_LINK ) {
			result = 0;
			return true;
		} else {
			printf("NULL head with a non-NULL tail\n");
			return false;
		}
	} else if ( tail == NULL_LINK ){
		printf("Non-NULL head with a NULL tail\n");
		return false;
	}

	ctx_link_adr iter = head;
	while(iter != NULL_LINK){
		if(host_arena[iter].meta_data.data != 0){
			printf("Link re-visited\n");
			ctx_link_adr loop_point = iter;
			ctx_link_adr visit_count = 0;
			iter = head;
			printf("(%d,%d): ",head,tail);
			ctx_link_adr step_count = 0;
			while(true){
				if(iter == loop_point){
					if(visit_count == 0){
						printf("{%d}->",iter);
					} else {
						printf("{%d}\n");
						break;
					}
					visit_count += 1;
				} else {
					printf("%d->",iter);
				}
				iter = host_arena[iter].next;
				step_count +=1;
				if(step_count > 64){
					printf("...{%d}\n",loop_point);
					break;
				}
			}
			return false;
		} else {
			host_arena[iter].meta_data.data = 1;
		}
		last = iter;
		iter = host_arena[iter].next;
		count += 1;
	}
	
	if( last != tail ){
		printf("Final link %d in the queue (%d,%d) not the tail\n",last,head,tail);
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
bool runtime_overview(runtime_context runtime){

	bool result = true;
	
	#ifdef DEBUG_PRINT
	const bool always_print = true;
	#else
	const bool always_print = true;
	#endif

	ctx_link_adr* pool_counts  = new ctx_link_adr[POOL_SIZE];
	bool*         pool_count_validity  = new bool[POOL_SIZE];
	ctx_link_adr* stack_counts = new ctx_link_adr[STACK_SIZE*QUEUES_PER_FRAME];
	bool*         stack_count_validity = new bool[STACK_SIZE*QUEUES_PER_FRAME];


	ctx_link* host_arena = new ctx_link[ARENA_SIZE];

	ctx_padded_queue* host_pool = new ctx_padded_queue[POOL_SIZE];

	ctx_stack* host_stack = new ctx_stack;


	ctx_link_adr link_total = 0;

	cudaMemcpy(host_arena,runtime.arena,sizeof(ctx_link) *ARENA_SIZE,cudaMemcpyDeviceToHost);
	cudaMemcpy(host_pool ,runtime.pool ,sizeof(ctx_queue)*POOL_SIZE ,cudaMemcpyDeviceToHost);
	cudaMemcpy(host_stack,runtime.stack,sizeof(ctx_stack)           ,cudaMemcpyDeviceToHost);


	for(ctx_link_adr i = 0; i < ARENA_SIZE; i++){
		host_arena[i].meta_data.data = 0;
	}



	//printf("Counting through pool links...\n");
	for(int i=0; i < POOL_SIZE; i++){
		//printf("Counting pool queue %d\n",i);	
		ctx_queue queue = host_pool[i].queue;
		pool_count_validity[i] = queue_count(host_arena,queue,pool_counts[i]);
		result = result && pool_count_validity[i];
		if(pool_count_validity[i]){
			link_total += pool_counts[i];
		}
	}

	//printf("Counting through stack links...\n");
	for(int i=0; i < STACK_SIZE; i++){
		for(int j=0; j < QUEUES_PER_FRAME; j++){
			ctx_queue queue = host_stack->frames[i].queues[j].queue;
			unsigned int index = i*QUEUES_PER_FRAME + j;
			stack_count_validity[i] = queue_count(host_arena,queue,stack_counts[index]);
			result = result && stack_count_validity[i];
			if(stack_count_validity[i]){
				link_total += stack_counts[index];
			}
		}

	}


	if( (!result) || always_print ){
		printf("POOL:\t[");
		for(int i=0; i<POOL_SIZE; i++){
			if(pool_count_validity[i]){
				printf("\t%d",pool_counts[i]);
			} else {
				printf("\t????");
			}
		}
		printf("\t]\n");

		unsigned int event_com	= host_stack->event_com;
		unsigned int depth	= (host_stack->depth_live >> 16) & 0xFFFF;
		unsigned int live	= (host_stack->depth_live) & 0xFFFF;
		printf("STACK:\t(event_com: %#010x\tdepth: %d\tlive: %d)\t{\n",event_com,depth,live);
		for(int i=0; i < STACK_SIZE; i++){
			unsigned int children_residents = host_stack->frames[i].children_residents;
			unsigned int children  = (children_residents >> 16) & 0xFFFF;
			unsigned int residents = children_residents & 0xFFFF;
			printf("(children: %d\t residents: %d)\t[",children,residents);
			for(int j=0; j < QUEUES_PER_FRAME; j++){
				unsigned int index = i*QUEUES_PER_FRAME + j;
				if(stack_count_validity[i]){
					printf("\t%d",stack_counts[index]);
				} else {
					printf("\t????");
				}
			}
			printf("\t]\n");

		}
		printf("} LINK TOTAL: %d\n",link_total);
	}

	delete[] host_arena;
	delete[] host_pool;
	delete[] host_stack;

	return result;

}


/*
__global__ void basic_collaz(unsigned int start, unsigned int end){

	for(unsigned int offset=blockIdx.x+start; offset < end; offset+= WG_COUNT){

		unsigned int original = offset+threadIdx.x;
		unsigned int val = original;
		unsigned int steps = 0;
		while(val > 1){
			if( (val % 2) == 0 ){
				val = val / 2;
			} else {
				val = val * 3 + 1;
			}
			steps++;
		}
		step_counts[original] = steps;


	}

}
*/



struct program_context;


program_context* initialize(runtime_context context, int argc, char *argv[]);


void finalize(runtime_context,program_context*);



int main(int argc, char *argv[]) {


	runtime_context runtime;

	/*
	// Allocate the arena, which is the range of memory that runtime links inhabit.
	*/
	cudaMalloc( (void**) &runtime.arena, ARENA_SIZE * sizeof(ctx_link)     );

	/*
	// Allocate the pool, which is an array of queues from which links are allocated.
	*/
	cudaMalloc( (void**) &runtime.pool , POOL_SIZE  * sizeof(ctx_padded_queue) );

	/*
	// Allocate the stack, which is an array of frames which hold links with thunk data.
	*/
	cudaMalloc( (void**) &runtime.stack, sizeof(ctx_stack) );	


	//printf("Initializing...\n");


	/* Zero out the runtime data structures. */
	init_runtime<<<WG_COUNT,WG_SIZE>>>(runtime);

	cudaDeviceSynchronize();
	checkError();
	

	/*
	printf("\n\n\n\nInitialization complete\n\n\n\n");
	runtime_overview(runtime);
	// */

	program_context* program = initialize(runtime,argc,argv);

	/*
	cudaDeviceSynchronize();
	checkError();
	runtime_overview(runtime);
	// */

	/* Limit for execution cycles per dispatch. */
	const unsigned int cycle_limit = 0xFFFFF;

	/* Limit for dispatch iterations per program lifetime. */
	const unsigned int iter_limit = 1;
	unsigned int iter_count = 0;

	/* Halting and communication events can be loaded into this field. */
	unsigned int event_com = 0;


	//printf("Executing...\n");

	//#define BASIC	
	while(true){
		
		/*
		// This is here to make sure accidental infinite loops encountered. This limit
		// on dispatches will likely be removed in practical execution.
		*/
		if(iter_count >= iter_limit){
			break;
		}
		

		/* Perform a round of execution. */
		exec_runtime<<<WG_COUNT,WG_SIZE>>>(runtime,cycle_limit);
		checkError();
		/* Load event signals from the GPU. */
		cudaMemcpyAsync(&event_com,&(runtime.stack->event_com),sizeof(unsigned int),cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		checkError();
		iter_count++;
		
		/* If a halting condition is reached, do not perform any more dispatches. */
		if(event_com != 0){
			break;
		}
	}

	finalize(runtime,program);
	
	/* Free our resources. */
	cudaFree(runtime.arena);
	cudaFree(runtime.pool);
	cudaFree(runtime.stack);

	return 0;	

}




