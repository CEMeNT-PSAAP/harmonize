


#ifndef HARMONIZE_SYNC_BARRIER
#define HARMONIZE_SYNC_BARRIER

#include "../async/mod.h"

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
			printf("\n\n\nVERY BAD\n\n\n");
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


#endif


