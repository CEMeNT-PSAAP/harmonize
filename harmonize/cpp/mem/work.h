#ifndef HARMONIZE_MEM_WORK
#define HARMONIZE_MEM_WORK


#include "../async/mod.h"




//!
//! The `WorkLink` template struct, given a `PromiseUnion` union, an address type, and a group
//! size, stores an array of `GROUP_SIZE` promise unions of the corresponding type and an
//! address value of type `ADR_TYPE`. Instances of this template also contain a `Op` value to
//! identify what type of work is contained within the link, a `meta_data` field, and a `count`
//! field to indicate the number of contained promises.
//!
template <typename OP_SET, typename ADR_TYPE, size_t GROUP_SIZE>
struct WorkLink
{

	using OpSet     = OP_SET;
	using AdrType   = ADR_TYPE;
	using UnionType = PromiseUnion<OpSet>;

	UnionType      promises[GROUP_SIZE];

	AdrType        next;
	unsigned int   meta_data;
	unsigned int   count;
	OpDisc         id;


	//! Zeros out a link, giving it a promise count of zero, a null function ID, and sets next
	//! to the given input.
	__host__ __device__ void empty(AdrType next_adr){

		next	= next_adr;
		id	= UnionType::Info::COUNT;
		count	= 0;

	}



	template<typename OP_TYPE>
	__device__ static constexpr Promise<OP_TYPE> promise_guard(Promise<OP_TYPE> promise) {
		static_assert(
			UnionType::template Lookup<OP_TYPE,OpSet>::type::CONTAINED,
			"\n\nTYPE ERROR: Type of promise cannot be contained in atomic work link.\n\n"
			"SUGGESTION: Double-check the signature of the atomic work link to make sure "
			"its OpUnion template parameter contains the desired operation type.\n\n"
			"SUGGESTION: Double-check the type signature of the promise and make sure "
			"it is the correct operation type.\n\n"
		);
		return promise;
	}

	//! Appends to the `WorkLink` by an atomic addition to the count field. This is only
	//! safe if it is already known that the `WorkLink` will have enough space to begin
	//! with.
	template<typename OP_TYPE>
	__device__ bool atomic_append(Promise<OP_TYPE> promise) {
		unsigned int index = atomicAdd(&count,1);
		promises[index] = promise_guard(promise);
		return (index == GROUP_SIZE);
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


typedef unsigned int PromiseCount;
typedef typename util::mem::PairEquivalent<PromiseCount>::Type PromiseCountPair;
//typedef unsigned long long int PromiseCountPair;

//! The `WorkFrame` template struct accepts a queue type and and a `size_t`, which is used to
//! define its iternal work pool. A `WorkFrame` represents a pool that tracks the current number
//! of contained promises as well as the number of "child" promises that could eventually return
//! to the frame.
template <typename QUEUE_TYPE, size_t QUEUE_COUNT>
struct WorkFrame
{

	util::mem::PairPack<PromiseCount> children_residents;
	WorkPool<QUEUE_TYPE, QUEUE_COUNT> pool;

};



//! The `WorkStack` template struct represents a series of `WorkFrames` following a heirarchy
//! of call depths. Currently, only a `STACK_SIZE` of zero is supported.
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

#endif


