

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




// A class used to manage lock-free remapping of promises
// in shared memory, defined generically over all programs. 
template<typename PROGRAM>
class SharedRemappingCheckpoint {

	using ProgramType = PROGRAM;
	using LinkType    = typename ProgramType::LinkType;
	using LinkAdrType = typename ProgramType::LinkAdrType;
	static const size_t GROUP_SIZE = ProgramType::GROUP_SIZE;

	using OpSet       = typename ProgramType::OpSet;
	using AdrType     = typename ProgramType::AdrType;
	using UnionType   = typename ProgramType::UnionType;

	using PairPack    = util::mem::PairPack<AdrType>;
	using PairType    = typename PairPack::PairType;

	//! The number of operations contained within the barrier's
	//! operation set.
	static const size_t TYPE_COUNT = UnionType::Info::COUNT;

	//! A table used to mediate the coalescing of links
	PairPack partial_table[UnionType::Info::COUNT];

	// The strategy currently used for a transaction
	enum TransactionMode {
		LAZY,
		OPTIMISTIC,
		PESSIMISTIC,
		COMPLETE
	};

	// Tracks the state of a single thread's transaction with
	// the remapping checkpoint.
	template<typename OP_TYPE>
	struct TransactionState {
		TransactionMode  mode;
		Promise<OP_TYPE> promise;
		PairPack         fullest;
		PairPack         emptiest;
	};
	
	// Ensure default construction is an option
	SharedRemappingCheckpoint<ProgramType>() = default;


	// Checks at compile time whether or not a given promise is valid for the
	// associated operation set
	template<typename OP_TYPE>
	__device__ static constexpr Promise<OP_TYPE> promise_guard(Promise<OP_TYPE> promise) {
		static_assert(
			UnionType::template Lookup<OP_TYPE,OpSet>::type::CONTAINED,
			"\n\nTYPE ERROR: Type of promise cannot be remapped at this checkpoint.\n\n"
			"SUGGESTION: Double-check the signature of the checkpoint to make sure "
			"its OpUnion template parameter contains the desired operation type.\n\n"
			"SUGGESTION: Double-check the type signature of the promise and make sure "
			"it is the correct operation type.\n\n"
		);
		return promise;
	}

	// Adds a single promise to the link addressed by the `dst` PairPack within the 
	template<typename OP_TYPE>
	__device__ bool add_promise(ProgramType &program, PairPack &dst, Promise<OP_TYPE> promise) {
		AdrType dst_index   = dst.get_left ();
		AdrType dst_address = dst.get_right();
		
		unsigned int total = dst_index + 1;
		unsigned int dst_total = ( total >= GROUP_SIZE ) ? GROUP_SIZE : total;
		unsigned int dst_delta = dst_total - dst_index;
		LinkType& dst_link  = program._grp_ctx.stash[dst_address];
		unsigned int dst_offset = atomicAdd(&dst_link.count,dst_delta);
		
		dst_link.promises[dst_offset] = promise;

		unsigned int checkout_delta = dst_delta;
		AdrType checkout = atomicAdd(&dst_link.next.adr,-checkout_delta);
		//printf("{[%d,%d]: %d checkout(%d->%d)}",blockIdx.x,threadIdx.x,dst_address,checkout,checkout-checkout_delta);

		//! If checkout==0, this thread is the last thread to modify the link, and the
		//! link has not been marked for dumping. This means we have custody of the link
		//! and must manage getting it into the queue via a partial slot or the full list.
		if( (checkout-checkout_delta) == 0 ){
			//printf("[%d-%d]: Got custody of link @%d for re-insertion.\n",blockIdx.x,threadIdx.x,dst_address);
			unsigned int final_count = atomicAdd(&dst_link.count,0);
			dst.set_left(final_count);
			unsigned int reset_delta = (GROUP_SIZE-final_count);
			unsigned int old_next = atomicAdd(&dst_link.next.adr,reset_delta);
			//printf("[%d-%d]: Reset the next field (%d->%d) of link @%d.\n",blockIdx.x,threadIdx.x,old_next,old_next+reset_delta,dst_address);
			__threadfence();
			return true;
		}
		//! If checkout==(GROUP_SIZE+1), this thread is the last thread to modify the
		//! link, and the link has been marked for dumping. This means we have custody
		//! of the link and must release it.
		else if ( (checkout-checkout_delta) == (GROUP_SIZE+1) ) {
			//printf("[%d-%d]: Got custody of link @%d for immediate dumping.\n",blockIdx.x,threadIdx.x,dst_address);
			atomicExch(&dst_link.next.adr,LinkAdrType::null);



			//!! TODO: Implement release

			return false;
		}
		//! In all other cases, we have no custody of the link.
		else {
			//printf("(checkout-checkout_delta) = %d\n",checkout-checkout_delta);
			__threadfence();
			return false;
		}

	}


	//! Attempt to add lazily
	template<typename OP_TYPE>
	__device__ void lazy_remap(ProgramType &program, TransactionState<OP_TYPE> &state) {

		OpDisc disc = UnionType::template Lookup<OP_TYPE>::type::DISC;

		//! Get the promise
		PairPack& part_slot = partial_table[disc];
		//! Find the value needed to bump the index by one
		PairType inc_val = PairPack::RIGHT_MASK + 1;
		//! Represents the link we'll be inserting into
		PairPack dst_pair;
		//! Attempt to claim a slot
		dst_pair.data = atomicAdd(&part_slot.data,inc_val);

		//! Recover the index/address info
		AdrType dst_index   = dst_pair.get_left ();
		AdrType dst_address = dst_pair.get_right();

		//! Check that the link address is valid and that the index is not overrun
		if ( (dst_address == LinkAdrType::null) || (dst_index >= GROUP_SIZE) ){
			//! No luck, there is no link to claim a slot in
			if((dst_index%GROUP_SIZE) == 0) {
				//! If the index is divisible by the link size, the thread is the
				//! designated allocator for the next link
				state.mode = TransactionMode::OPTIMISTIC;
			}
		} else if(add_promise(program, dst_pair, state.promise)) {
			//! The thread got custody, and must handle the resulting link.
			if(dst_pair.get_left() == GROUP_SIZE) {
				//! If full, the link can simply be queued
				state.mode = TransactionMode::COMPLETE;
			} else {
				//! If not full, more work must be done
				state.mode = TransactionMode::OPTIMISTIC;
			}
		} else {
			//! The thread did not get custody, so it's job is complete
			state.mode = TransactionMode::COMPLETE;
		}

	}


	template<typename OP_TYPE>
	__device__ void optimistic_remap(ProgramType& program, TransactionState<OP_TYPE>& state) {
		// Attempt to add lazily
	}

	template<typename OP_TYPE>
	__device__ void pessimistic_remap(ProgramType& program, TransactionState<OP_TYPE>& state) {
		// Attempt to add lazily
	}


	public:

	template<typename OP_TYPE>
	__device__ void remap(ProgramType& program, Promise<OP_TYPE> promise) {

		// Ensure the promise is valid
		promise = promise_guard(promise);

		// Initiate the transaction as lazy
		TransactionState<OP_TYPE> state = {
			TransactionMode::LAZY,
			promise,
			{0,LinkAdrType::null},
			{0,LinkAdrType::null}
		};

		// While the transaction is lazy, it should act lazy
		while( state.mode == TransactionMode::LAZY ) {
			lazy_remap(program,state);
		}

		// If laziness worked, dump any links that the thread
		// got custody of and return early.
		if(state.mode == TransactionMode::COMPLETE) {
			if(state.fullest.get_right() != LinkAdrType::null) {
				//! TODO: Full link dumping
			}
			return;	
		}

		while(state.mode != TransactionMode::COMPLETE){
			
			// Perform the designated strategy
			if (state.mode == TransactionMode::OPTIMISTIC) {
				optimistic_remap(program,state);
			} else {
				pessimistic_remap(program,state);
			}

			bool has_emptiest = (state.emptiest.get_right() != LinkAdrType::null);
			bool has_fullest  = (state.fullest .get_right() != LinkAdrType::null);
			// If there is an emptiest link that is truly empty, get rid of it
			if(has_emptiest && (state.emptiest.get_left() == 0)) {
				//! TODO: Empty link dumping
				state.emptiest = {0,LinkAdrType::null};
			}
			// If there is a fullest link that is truly full, get rid of it
			if(has_fullest && (state.fullest.get_left() == GROUP_SIZE)) {
				//! TODO: Full link dumping
				state.fullest = state.emptiest;
			}
			
			// Re-assess after dumping. If no fullest link exists, the 
			// transaction has concluded, with no links left over.
			if(state.fullest .get_right() != LinkAdrType::null){
				state.mode = TransactionMode::COMPLETE;
			}

		}

	}


};














//!
//! A `RemappingBarrier` coaleces promises into links in a lock-free manner.
//! Once released, all work in the queue is made available for work groups to
//! execute and all further appending operations will redirect promises to execution.
//! After being released, a queue may be reset to a non-released state.
//!
template<typename OP_SET, typename ADR_TYPE>
struct RemappingBarrier {


	#if 0
	using AdrType     = typename PROGRAM::AdrType;
	using LinkAdrType = typename PROGRAM::LinkAdrType;
	using QueueType   = typename PROGRAM::QueueType;
	using ProgramType = PROGRAM;
	using UnionType   = PromiseUnion<OP_SET>;
	using LinkType    = typename PROGRAM::LinkType;

	using PairPack    = util::mem::PairPack<AdrType>;
	using PairType    = typename PairPack::PairType;
	#else
	using AdrType     = ADR_TYPE;
	using LinkAdrType = util::mem::Adr<AdrType>;
	using QueueType   = util::mem::PoolQueue<LinkAdrType>;
	using UnionType   = PromiseUnion<OP_SET>;

	using PairPack    = util::mem::PairPack<AdrType>;
	using PairType    = typename PairPack::PairType;
	#endif

	//! The number of operations contained within the barrier's
	//! operation set.
	static const size_t TYPE_COUNT = UnionType::Info::COUNT;

	//! A tagged semaphore that communicates both the release state of the
	//! barrier, but the type of the barrier as well.
	TaggedSemaphore semaphore;
	unsigned int count; 

	//! A queue that contains all full links created by coalescing promises
	//! awaiting the release of the barrier.
	QueueType full_list;

	//! A table used to mediate the coalescing of links
	PairPack partial_table[UnionType::Info::COUNT];


	RemappingBarrier<OP_SET,ADR_TYPE>() = default;

	//! Creates a new `RemappingBarrier` with an empty full queue and partial table
	//! and with a semaphore value initialized to the supplied value.
	static __host__ __device__ RemappingBarrier<OP_SET,ADR_TYPE> blank(unsigned int sem_val)
	{
		RemappingBarrier<OP_SET,ADR_TYPE> result;
		result.semaphore = TaggedSemaphore(sem_val,1);
		result.count  = 0u;
		result.full_list.pair.data = QueueType::null;
		for(int i=0; i<TYPE_COUNT; i++){
			result.partial_table[i] = PairPack(0,LinkAdrType::null);
		}
		return result;
	}


	//! Counts the number of links in the given queue. This is only used for debugging purposes.
	template<typename PROGRAM>
	__device__ unsigned int queue_count(PROGRAM program, QueueType queue) {
		using ProgramType = PROGRAM;
		using LinkType    = typename ProgramType::LinkType;

		unsigned int total = 0;
		LinkAdrType iter = queue.get_head();
		while( ! iter.is_null() ){
		       LinkType& the_link = program._dev_ctx.arena[iter.adr];
		       total += the_link.count;
		       //printf("( [%d,%d] @%d #%d -> %d )",blockIdx.x,threadIdx.x,iter.adr,the_link.count,the_link.next.adr);
		       iter = the_link.next;
		}
		return total;
	}


	//! Releases a queue for execution.
	template<typename PROGRAM>
	__device__ void release_queue(PROGRAM program, QueueType queue, AdrType release_count) {
		using ProgramType = PROGRAM;
		using LinkType    = typename ProgramType::LinkType;
		__threadfence();
		//unsigned int true_count = queue_count(program,queue);
		//printf("[%d,%d]: Releasing queue (%d,%d) with count %d with delta %d\n",blockIdx.x,threadIdx.x,queue.get_head().adr,queue.get_tail().adr,true_count,release_count);
		unsigned int index = util::random_uint(program._thd_ctx.rand_state)%ProgramType::FRAME_SIZE;
		program.push_promises(0, index, queue, release_count);
	}


	//! This does not truely release partial links, but will mark the link for dumping
	//! by bumping the next semaphore by the index of the claimed pair plus 1. This means
	//! that, after all true promise insertions have occured, the next semaphore will
	//! be at GROUP_SIZE+1, a normally impossible value. Should the semaphore reach
	//! GROUP_SIZE+1 directly after the incrementation (and should the semaphore not
	//! have an original value of zero), this call will truely release the link. This
	//! caveat to zero-valued initial values prevents double-queuing.
	template<typename PROGRAM>
	__device__ void release_partial(PROGRAM program, PairPack pair) {
		using ProgramType = PROGRAM;
		using LinkType    = typename ProgramType::LinkType;
		static const size_t GROUP_SIZE = ProgramType::GROUP_SIZE;
		
		AdrType index   = pair.get_left ();
		AdrType address = pair.get_right();

		if( (address == LinkAdrType::null) || (index >= GROUP_SIZE) ){
			//printf("[%d,%d]: Tried to mark invalid pair (%d,@%d) for release\n",blockIdx.x,threadIdx.x,index,address);
			return;
		}
		

		if( index == 0 ){
			program.dump_spare_link(address);
		}

		LinkType& link = program._dev_ctx.arena[address];
		unsigned int delta = index+1;
		unsigned int checkout = atomicAdd(&link.next.adr,delta);
		
		//printf("[%d,%d]: Marking pair (%d,@%d) for release. delta(%d->%d)\n",blockIdx.x,threadIdx.x,index,address,checkout,checkout+delta);

	       	if( ( (checkout+delta) == (GROUP_SIZE+1) ) && (checkout != 0) ) {
			//printf("[%d,%d]: Instant release of (%d,@%d)\n",blockIdx.x,threadIdx.x,index,address);
			atomicExch(&link.next.adr,LinkAdrType::null);
			QueueType queue(address,address);
			unsigned int total = atomicAdd(&link.count,0);
			release_queue(program,queue,total);
		}
		

	}


	//! Releases the links in the full list.
	template<typename PROGRAM>
	__device__ void release_full(PROGRAM program) {
		using ProgramType = PROGRAM;
		using LinkType    = typename ProgramType::LinkType;
		
		QueueType queue;
		queue.pair.data  = atomicExch(&full_list.pair.data,QueueType::null);
		AdrType dump_count    = atomicExch(&count,0);
		__threadfence();
		if( (dump_count != 0) || (! queue.is_null() ) ){
			release_queue(program,queue,dump_count);
		}
		

	}


	//! Sweeps through full list and partial slots, releasing any queues or links that
	//! is found in the sweep.
	template<typename PROGRAM>
	__device__ void release_sweep(PROGRAM program) {	
		using ProgramType = PROGRAM;
		using LinkType    = typename ProgramType::LinkType;

		//printf("[%d,%d]: Performing release\n",blockIdx.x,threadIdx.x);	
		release_full(program);

		for( size_t i=0; i<TYPE_COUNT; i++ ) {
			PairPack null_pair(0,LinkAdrType::null);
			PairPack swap;
			swap.data = atomicExch(&partial_table[i].data,null_pair.data);
			release_partial(program,swap);
		}
	}


	//! Appends a full link at the given address to the full list.
	template<typename PROGRAM>
	__device__ void append_full(PROGRAM program, LinkAdrType address){
		using ProgramType = PROGRAM;
		using LinkType    = typename ProgramType::LinkType;
		static const size_t GROUP_SIZE = ProgramType::GROUP_SIZE;
		//printf("Appending full link @%d\n",address.adr);
		

		if( ! address.is_null()	) {
			LinkType& dst_link = program._dev_ctx.arena[address.adr];
			unsigned int count = atomicAdd(&dst_link.count,0);
			//printf("[%d,%d]:Appending full link @%d with count %d\n",blockIdx.x,threadIdx.x,address.adr,count);
		}
		
		atomicAdd(&count,GROUP_SIZE);
		program._dev_ctx.arena[address].next.adr = LinkAdrType::null;
		QueueType src_queue = QueueType(address,address);
		program.push_queue(full_list,src_queue);
	}


	//! 
	//! Merges data from the link supplied as the third argument into the remaning space
	//! available in the link supplied as the second argument. The link given by the third
	//! argument must never have any merges in flight with it as the destination. The link
	//! given by the second argument may have concurent merges in flight with it as the 
	//! destination. If the link given by the second argument was claimed from a partial
	//! slot through an atomic exchange. If it has been claimed, the fourth argument MUST be
	//! true. Likewise, if it has not been claimed, the fourth argument MUST NOT be true.
	//! This is done to ensure that, if a link is merged into but is still not full, that
	//! link can be safely used in future merging operations as a source.
	//!
	//! After merging, the count fields of the input pairs are updated to reflect the change
	//! in occupancy. If the current thread is found to have custody of the destination
	//! link, this function returns true. Otherwise, false.
	//! 
	template<typename PROGRAM>
	__device__ bool merge_links(PROGRAM program, PairPack& dst, PairPack& src, bool claimed){
		using ProgramType = PROGRAM;
		using LinkType    = typename ProgramType::LinkType;
		static const size_t GROUP_SIZE = ProgramType::GROUP_SIZE;
	
		AdrType dst_index   = dst.get_left ();
		AdrType dst_address = dst.get_right();
		AdrType src_index   = src.get_left ();
		AdrType src_address = src.get_right();
		
		if( src_index >= GROUP_SIZE ){
			printf("Error: Link merge encountered. src_index >= GROUP_SIZE");
		}

		unsigned int total = dst_index + src_index;

		unsigned int dst_total = ( total >= GROUP_SIZE ) ? GROUP_SIZE         : total;
		unsigned int src_total = ( total >= GROUP_SIZE ) ? total - GROUP_SIZE :     0;

		unsigned int dst_delta = dst_total  - dst_index;
		
		//printf("{[%d,%d]: src is (%d,@%d) }",blockIdx.x,threadIdx.x,src_index,src_address);

		LinkType& dst_link  = program._dev_ctx.arena[dst_address];

		//if( claimed ) {
			//unsigned int old_next = atomicAdd(&dst_link.next.adr,(AdrType)-dst_empty);
			//printf("{[%d,%d]: claimed @%d}",blockIdx.x,threadIdx.x,dst_address);
		//}

		unsigned int dst_offset = atomicAdd(&dst_link.count,dst_delta);
		unsigned int src_offset = src_index - dst_delta;
		
		//printf("{[%d,%d]: count(%d->%d)}",blockIdx.x,threadIdx.x,dst_offset,dst_offset+dst_delta);

		//printf("%d: src_offset=%d\tdst_offset=%d\tdst_delta=%d\n",blockIdx.x,src_offset,dst_offset,dst_delta);

		if(src_address != LinkAdrType::null){
			LinkType& src_link  = program._dev_ctx.arena[src_address];
			for(unsigned int i = 0; i < dst_delta; i++){
				dst_link.promises[dst_offset+i] = src_link.promises[src_offset+i];
			}
			atomicAdd(&src_link.count   ,-dst_delta);
			atomicAdd(&src_link.next.adr, dst_delta);
		}		

		unsigned int checkout_delta = dst_delta;
		if( claimed ){
			checkout_delta += GROUP_SIZE - dst_total;
		}
		AdrType checkout = atomicAdd(&dst_link.next.adr,-checkout_delta);
		//printf("{[%d,%d]: %d checkout(%d->%d)}",blockIdx.x,threadIdx.x,dst_address,checkout,checkout-checkout_delta);
		src.set_left(src_total);

		//! If checkout==0, this thread is the last thread to modify the link, and the
		//! link has not been marked for dumping. This means we have custody of the link
		//! and must manage getting it into the queue via a partial slot or the full
		//! list.
		if( (checkout-checkout_delta) == 0 ){
			//printf("[%d-%d]: Got custody of link @%d for re-insertion.\n",blockIdx.x,threadIdx.x,dst_address);
			unsigned int final_count = atomicAdd(&dst_link.count,0);
			dst.set_left(final_count);
			unsigned int reset_delta = (GROUP_SIZE-final_count);
			unsigned int old_next = atomicAdd(&dst_link.next.adr,reset_delta);
			//printf("[%d-%d]: Reset the next field (%d->%d) of link @%d.\n",blockIdx.x,threadIdx.x,old_next,old_next+reset_delta,dst_address);
			__threadfence();
			return true;
		}
		//! If checkout==(GROUP_SIZE+1), this thread is the last thread to modify the
		//! link, and the link has been marked for dumping. This means we have custody
		//! of the link and must release it.
		else if ( (checkout-checkout_delta) == (GROUP_SIZE+1) ) {
			//printf("[%d-%d]: Got custody of link @%d for immediate dumping.\n",blockIdx.x,threadIdx.x,dst_address);
			atomicExch(&dst_link.next.adr,LinkAdrType::null);
			QueueType queue(dst_address,dst_address);
			unsigned int total = atomicAdd(&dst_link.count,0);
			__threadfence();
			release_queue(program,queue,total);
			__threadfence();
			return false;
		}
		//! In all other cases, we have no custody of the link.
		else {
			//printf("(checkout-checkout_delta) = %d\n",checkout-checkout_delta);
			__threadfence();
			return false;
		}

	}


	//! Awaits the barrier with a promise union, using the provided discriminant to determine
	//! the type of the contained promise.
	template<bool CAN_RELEASE=true, typename PROGRAM>
	__device__ void union_await(PROGRAM program, OpDisc disc, typename PROGRAM::PromiseUnionType promise_union) {
		using ProgramType = PROGRAM;
		using LinkType    = typename ProgramType::LinkType;
		static const size_t GROUP_SIZE = ProgramType::GROUP_SIZE;
		
		//printf("Performing atomic append for type with discriminant %d.\n",disc);
		
		PairPack& part_slot = partial_table[disc];
		PairType inc_val = PairPack::RIGHT_MASK + 1;
		bool optimistic = true;
		
		//! We check the semaphore. If the semaphore is non-zero, the queue has been
		//! released, and so the promise can be queued normally. This check does not
		//! force a load with atomics (as is done later) because the benefits of
		//! a forced load don't seem to make up for the overhead.
		if( (semaphore.sem == 0) && CAN_RELEASE ){
			//printf("early exit\n");
	 		program.async_call(disc, 0, promise_union);
			return;
		}

		//! We start off with a link containing just our input promise. Depending
		//! upon how merges transpire, this link will either fill up and be
		//! successfully deposited into the queue, or will have its contents
		//! drained into a different link and will be stored away for future use.
		LinkAdrType first_link_adr = program.alloc_spare_link();
		PairPack spare_pair(1u,first_link_adr.adr);
		LinkType&   start_link     = program._dev_ctx.arena[first_link_adr];
		start_link.id    = disc;
		start_link.count = 1;
		start_link.next  = GROUP_SIZE - 1;
		start_link.promises[0] = promise_union;
		__threadfence();

		unsigned int opt_fail_count = 0;
		bool had_custody = true;

		//printf("Allocated spare link %d.\n",first_link_adr.adr);

		while(true) {
			PairPack dst_pair;

			LinkAdrType spare_link_adr = spare_pair.get_right();
			LinkType& spare_link = program._dev_ctx.arena[spare_link_adr];

			//! Attempt to claim a slot in the link just by incrementing
			//! the index of the index/address pair pack. This is quicker,
			//! but only works if the incrementation reaches the field before
			//! other threads claim the remaining promise slots.
			if ( optimistic ) {
				//printf("Today, we choose optimisim.\n");
				dst_pair.data = atomicAdd(&part_slot.data,inc_val*spare_pair.get_left());
				//printf("{[%d,%d] Optimistic (%d->%d,@%d)}",blockIdx.x,threadIdx.x,dst_pair.get_left(),dst_pair.get_left()+spare_pair.get_left(),dst_pair.get_right());
			}
			//! Gain exclusive access to link via an atomic exchange. This is
			//! slower, but has guaranteed progress. 
			else {
				//printf("Today, we resort to pessimism.\n");
				PairPack null_pair(0,LinkAdrType::null);
				dst_pair.data = atomicExch(&part_slot.data,spare_pair.data);
				spare_pair = PairPack(0,LinkAdrType::null);
				//printf("{[%d,%d] Pessimistic (%d,@%d)}",blockIdx.x,threadIdx.x,dst_pair.get_left(),dst_pair.get_right());
			}

			
			AdrType dst_index   = dst_pair.get_left ();
			AdrType dst_address = dst_pair.get_right();
			//! Handle cases where represented link is null or already full
			if ( (dst_address == LinkAdrType::null) || (dst_index >= GROUP_SIZE) ){
				//! Optimistic queuing must retry, but pessimistic
				//! queueing can get away with leaving its link behind
				//! and doing nothing else.
				if ( optimistic ) {
					opt_fail_count += 1;
					optimistic = (dst_index % GROUP_SIZE) > opt_fail_count;
					continue;
				} else {
					break;
				}
			} else {
				bool owns_dst = merge_links(program,dst_pair,spare_pair,!optimistic);
				had_custody = owns_dst;
				optimistic = (dst_index % GROUP_SIZE) != 0;
				opt_fail_count = 0;
				//printf("owns_dst=%d\n",owns_dst);
				//! If the current thread has custody of the destination link,
				//! it must handle appending it to the full list if it is full and
				//! merging it into the partial slot if it is partial.
				if ( owns_dst ){
					//! Append full destination links to the full list.
					//! DO NOT BREAK FROM THE LOOP. There may be a partial
					//! source link that still needs to be merged in another
					//! pass.
					if ( dst_pair.get_left() == GROUP_SIZE ) {
						append_full(program,dst_address);
					}
					//! Dump the current spare link and restart the merging
					//! procedure with our new partial link
					else if ( dst_pair.get_left() != 0 ) {
						program.dump_spare_link(spare_pair.get_right());
						spare_pair = dst_pair;
						continue;
					}
					//! This case should not happen, but it does not hurt to 
					//! include a branch to handle it, in case something
					//! unexpected occurs.
					else {
						//printf("\n\nTHIS PRINT SHOULD BE UNREACHABLE\n\n");
						program.dump_spare_link(spare_pair.get_right());
						program.dump_spare_link(  dst_pair.get_right());
						break;
					}
				}
				//! If the spare link is empty, dump it rather than try to merge
				if (spare_pair.get_left() == 0) {
					program.dump_spare_link(spare_pair.get_right());
					break;
				}
			}

		}


		//! Double-check the sememaphore at the end. It is very important we do this double
		//! check and that we do it at the very end of every append operation. Because
		//! custody of partial links can be given to any append operation of the
		//! corresponding operation type, and we don't know which append operation
		//! comes last, we need to assume that, if the append has gotten this far, it
		//! may be the last append and hence should make sure that no work is left
		//! behind.
		__threadfence();
		unsigned int now_semaphore = atomicAdd(&semaphore.sem,0);
		if( (now_semaphore == 0) && had_custody && CAN_RELEASE ){
			//printf("[%d,%d]: Last-minute release required!\n",blockIdx.x,threadIdx.x);
			release_sweep(program);
		}


	}
	

	//! Awaits the barrier with th supplied promise.
	template<bool CAN_RELEASE=true, typename PROGRAM, typename OP_TYPE>
	__device__ void await(PROGRAM program, Promise<OP_TYPE> promise) {
		using ProgramType = PROGRAM;
		using LinkType    = typename ProgramType::LinkType;
		static const size_t GROUP_SIZE = ProgramType::GROUP_SIZE;
		
		//! Guards invocations of the function to make sure invalid promise types
		//! are not passed in.
		static_assert(
			UnionType::template Lookup<OP_TYPE>::type::CONTAINED,
			"TYPE ERROR: Remapping work queue cannot queue promises of this type."
		);

		OpDisc disc = UnionType::template Lookup<OP_TYPE>::type::DISC;
		
		//printf("Performing atomic append for type with discriminant %d.\n",disc);
		
		PairPack& part_slot = partial_table[disc];
		PairType inc_val = PairPack::RIGHT_MASK + 1;
		bool optimistic = true;
		
		//! We check the semaphore. If the semaphore is non-zero, the queue has been
		//! released, and so the promise can be queued normally. This check does not
		//! force a load with atomics (as is done later) because the benefits of
		//! a forced load don't seem to make up for the overhead.
		if( (semaphore.sem == 0) && CAN_RELEASE ){
			//printf("early exit\n");
	 		program.async_call_cast(0, promise);
			return;
		}

		//! We start off with a link containing just our input promise. Depending
		//! upon how merges transpire, this link will either fill up and be
		//! successfully deposited into the queue, or will have its contents
		//! drained into a different link and will be stored away for future use.
		LinkAdrType first_link_adr = program.alloc_spare_link();
		PairPack spare_pair(1u,first_link_adr.adr);
		LinkType&   start_link     = program._dev_ctx.arena[first_link_adr];
		start_link.id    = disc;
		start_link.count = 1;
		start_link.next  = GROUP_SIZE - 1;
		start_link.promises[0] = promise;
		__threadfence();

		unsigned int opt_fail_count = 0;
		bool had_custody = true;

		//printf("Allocated spare link %d.\n",first_link_adr.adr);

		while(true) {
			PairPack dst_pair;

			LinkAdrType spare_link_adr = spare_pair.get_right();
			LinkType& spare_link = program._dev_ctx.arena[spare_link_adr];

			//! Attempt to claim a slot in the link just by incrementing
			//! the index of the index/address pair pack. This is quicker,
			//! but only works if the incrementation reaches the field before
			//! other threads claim the remaining promise slots.
			if ( optimistic ) {
				//printf("Today, we choose optimisim.\n");
				dst_pair.data = atomicAdd(&part_slot.data,inc_val*spare_pair.get_left());
				//printf("{[%d,%d] Optimistic (%d->%d,@%d)}",blockIdx.x,threadIdx.x,dst_pair.get_left(),dst_pair.get_left()+spare_pair.get_left(),dst_pair.get_right());
			}
			//! Gain exclusive access to link via an atomic exchange. This is
			//! slower, but has guaranteed progress. 
			else {
				//printf("Today, we resort to pessimism.\n");
				PairPack null_pair(0,LinkAdrType::null);
				dst_pair.data = atomicExch(&part_slot.data,spare_pair.data);
				spare_pair = PairPack(0,LinkAdrType::null);
				//printf("{[%d,%d] Pessimistic (%d,@%d)}",blockIdx.x,threadIdx.x,dst_pair.get_left(),dst_pair.get_right());
			}

			
			AdrType dst_index   = dst_pair.get_left ();
			AdrType dst_address = dst_pair.get_right();
			//! Handle cases where represented link is null or already full
			if ( (dst_address == LinkAdrType::null) || (dst_index >= GROUP_SIZE) ){
				//! Optimistic queuing must retry, but pessimistic
				//! queueing can get away with leaving its link behind
				//! and doing nothing else.
				if ( optimistic ) {
					opt_fail_count += 1;
					optimistic = (dst_index % GROUP_SIZE) > opt_fail_count;
					continue;
				} else {
					break;
				}
			} else {
				bool owns_dst = merge_links(program,dst_pair,spare_pair,!optimistic);
				had_custody = owns_dst;
				optimistic = (dst_index % GROUP_SIZE) != 0;
				opt_fail_count = 0;
				//printf("owns_dst=%d\n",owns_dst);
				//! If the current thread has custody of the destination link,
				//! it must handle appending it to the full list if it is full and
				//! merging it into the partial slot if it is partial.
				if ( owns_dst ){
					//! Append full destination links to the full list.
					//! DO NOT BREAK FROM THE LOOP. There may be a partial
					//! source link that still needs to be merged in another
					//! pass.
					if ( dst_pair.get_left() == GROUP_SIZE ) {
						append_full(program,dst_address);
					}
					//! Dump the current spare link and restart the merging
					//! procedure with our new partial link
					else if ( dst_pair.get_left() != 0 ) {
						program.dump_spare_link(spare_pair.get_right());
						spare_pair = dst_pair;
						continue;
					}
					//! This case should not happen, but it does not hurt to 
					//! include a branch to handle it, in case something
					//! unexpected occurs.
					else {
						//printf("\n\nTHIS PRINT SHOULD BE UNREACHABLE\n\n");
						program.dump_spare_link(spare_pair.get_right());
						program.dump_spare_link(  dst_pair.get_right());
						break;
					}
				}
				//! If the spare link is empty, dump it rather than try to merge
				if (spare_pair.get_left() == 0) {
					program.dump_spare_link(spare_pair.get_right());
					break;
				}
			}

		}




		//! Double-check the sememaphore at the end. It is very important we do this double
		//! check and that we do it at the very end of every append operation. Because
		//! custody of partial links can be given to any append operation of the
		//! corresponding operation type, and we don't know which append operation
		//! comes last, we need to assume that, if the append has gotten this far, it
		//! may be the last append and hence should make sure that no work is left
		//! behind.
		__threadfence();
		unsigned int now_semaphore = atomicAdd(&semaphore.sem,0);
		if( (now_semaphore == 0) && had_custody && CAN_RELEASE ){
			//printf("[%d,%d]: Last-minute release required!\n",blockIdx.x,threadIdx.x);
			release_sweep(program);
		}


	}

	//! Sets the semaphore to zero and performs a release sweep. The sweep is necessary, because
	//! it is possible for no append operations to occur after a release operation.
	template<typename PROGRAM>
	__device__ void release(PROGRAM program) {
		unsigned int now_semaphore = atomicExch(&semaphore.sem,0);
		if( now_semaphore != 0 ) {
			release_sweep(program);
		}
	}

	//! Adds the supplied delta value to the semaphore and performs a release sweep if the
	//! result value is zero.
	template<typename PROGRAM>
	__device__ void add_semaphore(PROGRAM program,unsigned int delta) {
		unsigned int now_semaphore = atomicAdd(&semaphore.sem,delta);
		if( (now_semaphore+delta) == 0 ) {
			release_sweep(program);
		}
	}

	//! Subtracts the supplied delta value from  the semaphore and performs a release sweep
	//! if the result value is zero.
	template<typename PROGRAM>
	__device__ void sub_semaphore(PROGRAM program,unsigned int delta) {
		unsigned int now_semaphore = atomicAdd(&semaphore.sem,(unsigned int) -delta);
		if( (now_semaphore-delta) == 0 ) {
			release_sweep(program);
		}
	}

};




//! The `UnitBarrier` template struct acts as a barrier for a single await. This is useful for
//! setting up additional layers of resolution for multi-dependency awaits.
template<typename OP_SET, typename ADR_TYPE>
struct UnitBarrier {
	
	using AdrType = ADR_TYPE;

	TaggedSemaphore semaphore;
	PromiseEnum<OP_SET> promise;


	UnitBarrier<OP_SET,ADR_TYPE>() = default;

	UnitBarrier<OP_SET,ADR_TYPE>(PromiseEnum<OP_SET> promise_value, unsigned int semaphore_value)
		: semaphore(semaphore_value,0)
		, promise(promise_value)
	{}




	//! Sets the semaphore value to zero and, if it was not already zero, releases the
	//! promise for execution
	template<typename PROGRAM>
	__device__ void release(PROGRAM program) {
		unsigned int old_val = atomicExch(&semaphore.sem,0);
		if( old_val != 0 ) {
			program.async_call(promise);
		}
	}

	//! Adds the supplied delta value to the semaphore and performs a release sweep if the
	//! result value is zero and the previous value was not zero.
	template<typename PROGRAM>
	__device__ void add_semaphore(PROGRAM program,unsigned int delta) {
		unsigned int old_val = atomicAdd(&semaphore.sem,delta);
		if( ((old_val+delta) == 0)  && (old_val != 0) ) {
			program.async_call(promise);
		}
	}

	//! Subtracts the supplied delta value from  the semaphore and performs a release sweep
	//! if the result value is zero and the previous value was not zero.
	template<typename PROGRAM>
	__device__ void sub_semaphore(PROGRAM program,unsigned int delta) {
		unsigned int old_val = atomicAdd(&semaphore.sem,(unsigned int) -delta);
		if( ((old_val-delta) == 0) && (old_val != 0) ) {
			program.async_call(promise);
		}
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



//! A simple allocator which allocates positions in a series using atomic
//! bit operations. While not ideal for long series, the higher "surface area"
//! provided by the mask-based approach can reduce bottlenecking over linked
//! list based approaches when many agents are contending over a relatively
//! small pool of resources.
template <unsigned int SLOT_COUNT>
struct MaskAllocator {
	static size_t const ELEMENT_COUNT = (SLOT_COUNT+63) / 64;
	int total;
	int count;
	unsigned long long int mask_list[ELEMENT_COUNT];

	__device__ void set_total (int tot){
		total = tot;
	}

	__device__ void set_all (bool default_val){
		int val = 0;
		count   = 0;
		if (default_val) {
			total = SLOT_COUNT;
			val   = 0xFFFFFFFFFFFFFFFF;
			count = total;
		}
		for(int i=0; i<ELEMENT_COUNT; i++){
			mask_list[i] = val;
		}
	}

	__device__ int alloc () {
		int old_value = atomicSub(&count,1);
		if ( old_value <= 0 ) {
			atomicAdd(&count,1);
			return -1;
		}
		int index = -1;
		while (index == -1) {
			while(index == -1) {
				for (int i=0; i<ELEMENT_COUNT; i++){
					if (mask_list[i] == 0){
						continue;
					}
					index = (63-clz(mask_list[i])) + (i*64);
				}
			}
			unsigned int elem_idx = index / 64;
			unsigned int bit_idx  = index % 64;
			unsigned int mask =  ~(1 << bit_idx);
			unsigned int old = atomicAnd(mask_list+elem_idx,mask);
			if ( (old&mask) == old){
				index = -1;
			}
		}
		return index;
	}

	__device__ void free (int index) {
		unsigned int elem_idx = index / 64;
		unsigned int bit_idx  = index % 64;
		unsigned int mask =  1 << bit_idx;
		unsigned int old = atomicOr(mask_list+elem_idx,mask);
		if ( (old|mask) != old ) {
			atomicAdd(&count,1);
		}
	}


};


