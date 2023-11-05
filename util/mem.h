#pragma once

#if defined(__NVCC__) || HIPIFY
	#include "adapt.h"
	#include "basic.h"
	#include "host.h"
#elif defined(__HIP__)
	#include "adapt.h.hip"
	#include "basic.h.hip"
	#include "host.h.hip"
#endif



namespace mem {

template<typename T>
struct PairEquivalent;

template<>
struct PairEquivalent<unsigned short>
{
	typedef unsigned int Type;
};


template<>
struct PairEquivalent<unsigned int>
{
	typedef unsigned long long int Type;
};


template<typename T>
struct PairPack
{

	typedef PairPack<T> Self;

	typedef typename PairEquivalent<T>::Type PairType;

	static const PairType RIGHT_MASK = std::numeric_limits<T>::max();
	static const size_t   HALF_WIDTH = std::numeric_limits<T>::digits;
	static const PairType LEFT_MASK  = RIGHT_MASK << (PairType) HALF_WIDTH;

	PairType data;

	__host__  __device__ T    get_left() {
		return (data >> HALF_WIDTH) & RIGHT_MASK;
	}

	__host__  __device__ void set_left(T val) {
		data &= RIGHT_MASK;
		data |= ((PairType) val) << (PairType) HALF_WIDTH;
	}

	__host__  __device__ T    get_right(){
		return data & RIGHT_MASK;
	}

	__host__  __device__ void set_right(T val){
		data &= LEFT_MASK;
		data |= val;
	}

	PairPack<T> () = default;

	__host__  __device__ PairPack<T> (T left, T right){
		data   = left;
		data <<= HALF_WIDTH;
		data  |= right;
	}

};







template <typename ADR_INNER_TYPE>
struct Adr
{

	typedef ADR_INNER_TYPE AdrType;

	static const AdrType null = std::numeric_limits<AdrType>::max();

	AdrType adr;


	Adr<AdrType>() = default;

	__host__ __device__ Adr<AdrType>(AdrType adr_val) : adr(adr_val) {}

	__host__ __device__ bool operator==(const Adr<AdrType>& link_adr){
		return (adr == link_adr.adr);
	}


	 __host__ __device__ bool is_null(){
		return (adr == null);
	}


};





template <typename ADR_TYPE>
struct PoolQueue;


template <typename ADR_TYPE>
struct PoolQueue <Adr<ADR_TYPE>>
{

	typedef ADR_TYPE AdrType;
	typedef Adr<AdrType> LinkAdrType;
	typedef PoolQueue<LinkAdrType> Self;

	typedef typename PairEquivalent<AdrType>::Type QueueType;
	PairPack<AdrType> pair;

	static const QueueType null       = std::numeric_limits<QueueType>::max();

	__host__ __device__ LinkAdrType get_head(){
		return pair.get_left();
	}

	/*
	// This function extracts the tail of the given queue
	*/
	__host__ __device__ LinkAdrType get_tail(){
		return pair.get_right();
	}

	/*
	// Enable default constructor
	*/
	PoolQueue<Adr<ADR_TYPE>>() = default;

	/*
	// This function concatenates two link addresses into a queue
	*/
	__host__ __device__ PoolQueue<Adr<ADR_TYPE>>
	(LinkAdrType head, LinkAdrType tail)
	{
		pair = PairPack<AdrType>(head.adr,tail.adr);
	}

	/*
	// This function sets the head of the queue to the given link address
	*/
	__host__ __device__ void set_head(LinkAdrType head){
		pair.set_left(head.adr);
	}

	/*
	// This function sets the tail of the queue to the given link address
	*/
	__host__ __device__ void set_tail(LinkAdrType tail){
		pair.set_right(tail.adr);
	}

	/*
	// Returns true if and only if queue is empty
	*/
	 __host__ __device__ bool is_null(){
		return (pair.data == Self::null);
	}


	__host__ __device__ static const Self null_queue() {
		Self result;
		result.pair.data = Self::null;
		return result;
	}

};








template<typename T, typename AdrType>
__host__ void mempool_check(T* arena, size_t arena_size, PoolQueue<AdrType>* pool, size_t pool_size){




}








#define LAZY_MEM

#if defined(__NVCC__)
	//#define SMART_MEM
#endif


template<typename T, typename INDEX> struct MemPool;

template<typename T, typename INDEX>
__global__ void mempool_init(MemPool<T,INDEX> mempool);

template<typename T, typename INDEX>
struct MemPool {

	typedef T     DataType;
	typedef INDEX Index;
	typedef Adr<Index> AdrType;
	typedef PoolQueue<AdrType> QueueType;

	static const unsigned int RETRY_COUNT = 32;

	union Link {
		DataType data;
		AdrType  next;
	};

	#ifdef LAZY_MEM
	AdrType  claim_count;
	#endif

	Link*     arena;
	AdrType   arena_size;

	QueueType* pool;
	AdrType    pool_size;


	__host__ void host_init()
	{
		#ifdef LAZY_MEM
		claim_count = 0;
		#endif

		if( arena_size.adr != 0 ){
			arena  = host::hardMalloc<Link>     ( arena_size.adr );
		}
		if( pool_size.adr  != 0 ){
			pool   = host::hardMalloc<QueueType>( pool_size .adr );
		}
		if( (arena_size.adr != 0) || (pool_size.adr != 0 ) ){
			mempool_init<<<256,32>>>(*this);
			if ( cudaDeviceSynchronize() ) {
				printf("Failed to synchronize with host when initializing MemPool.\n");
			}
		}
		//next_print();
	}

	__host__ void host_free()
	{
		if( arena_size.adr != 0 ){
			host::auto_throw( cudaFree( arena ) );
		}
		if( pool_size.adr  != 0 ){
			host::auto_throw( cudaFree( pool  ) );
		}
	}


	__host__ void next_print(){
		Link* host_copy = (Link*)malloc(sizeof(Link)*arena_size.adr);
		cudaError_t copy_err = cudaMemcpy(host_copy,arena,sizeof(Link)*arena_size.adr,cudaMemcpyDeviceToHost);
		if ( copy_err != cudaSuccess ) {
			printf("Failed to memcpy in MemPool::next_print.\n");
		}
		for(int i=0; i<arena_size.adr; i++){
			printf("%d,",host_copy[i].next.adr);
		}
	}


	__host__ MemPool<DataType,Index>( Index as, Index ps )
		: arena_size(as)
		, pool_size (ps)
	{}

	__host__ MemPool<DataType,Index>()
		: arena_size(0)
		, pool_size (0)
	{}

	__device__ DataType& operator[] (Index index){
		return arena[index].data;
	}


	__device__ Index pop_front(QueueType& queue){
		AdrType result;
		if( queue.is_null() ){
			result.adr = AdrType::null;
		} else {
			result = queue.get_head();
			AdrType next = arena[result.adr].next;
			queue.set_head(next);
			if( next.is_null() ){
				queue.set_tail(next);
			} else if ( queue.get_tail() == result ){
				printf("ERROR: Final link does not have a null next.\n");
				queue.pair.data = QueueType::null;
				return result.adr;
			}
		}
		return result.adr;
	}


	__device__ QueueType join(QueueType dst, QueueType src){
		if( dst.is_null() ){
			return src;
		} else if ( src.is_null() ) {
			return dst;
		} else {
			AdrType left_tail_adr  = dst.get_tail();
			AdrType right_head_adr = src.get_head();
			AdrType right_tail_adr = src.get_tail();

			/*
			// Find last link in the dst queue and set succcessor to head of src queue.
			*/
			arena[left_tail_adr.adr].next = right_head_adr;

			/* Set the right half of the left_queue handle to index the new tail. */
			dst.set_tail(right_tail_adr);

			return dst;
		}
		//arena[dst.get_tail().adr].next = src.get_head();
		//dst.set_tail(src.get_tail());
		//return dst;
	}


	__device__ QueueType pull_queue(Index& pull_idx){
		Index start_idx = pull_idx;
		//printf("Pulling from %d",start_idx);
		bool done = false;
		QueueType queue = QueueType::null_queue();
		__threadfence();
		for(Index i=start_idx; i<pool_size.adr; i++){
			queue.pair.data = atomicExch(&(pool[i].pair.data),QueueType::null);
			if( ! queue.is_null() ){
				done = true;
				pull_idx = i;
				//printf("{pulled(%d,%d) from %d}",queue.get_head().adr,queue.get_tail().adr,pull_idx);
				return queue;
			}
		}
		if( !done ){
			for(Index i=0; i<start_idx; i++){
				queue.pair.data = atomicExch(&(pool[i].pair.data),QueueType::null);
				if( ! queue.is_null() ){
					done = true;
					pull_idx = i;
					//printf("{pulled(%d,%d) from %d}",queue.get_head().adr,queue.get_tail().adr,pull_idx);
					return queue;
				}
			}
		}
		//printf("{pulled(%d,%d)}",queue.get_head().adr,queue.get_tail().adr);
		return queue;
	}



	__device__ void push_queue(QueueType queue, Index push_idx){
		if( queue.is_null() ){
			return;
		}
		//printf("Pushing (%d,%d) into %d",queue.get_head().adr,queue.get_tail().adr,push_idx);
		unsigned int try_count = 0;
		while(true){
			__threadfence();
			QueueType swap;
			swap.pair.data = atomicExch(&(pool[push_idx].pair.data),queue.pair.data);
			if( swap.is_null() ){
				//printf("{%d tries}",try_count);
				return;
			}
			try_count++;
			queue.pair.data = atomicExch(&(pool[push_idx].pair.data),QueueType::null);
			queue = join(swap,queue);
		}
	}


	__device__ Index pull_span( Index* dst, Index count, Index stride, unsigned int& rand_state ){

		Index result = 0;
		for(unsigned int t=0; t<RETRY_COUNT; t++){
			Index queue_index = random_uint(rand_state) % pool_size.adr;
			QueueType queue = pull_queue(queue_index);
			if( ! queue.is_null() ){
				AdrType adr = pop_front(queue);
				while( (adr.adr != AdrType::null) && (result < count) ){
					dst[result*stride] = adr.adr;
					adr = pop_front(queue);
					result++;
				}
				push_queue(queue,queue_index);
			}
			if( result >= count ){
				return result;
			}
		}
		return result;
	}



	__device__ void push_span( Index* src, Index count, Index stride, unsigned int& rand_state ){
		Index first = AdrType::null;
		Index last  = AdrType::null;
		for( Index i=0; i<count; i++){
			Index& slot = src[stride*i];
			if( slot != AdrType::null ){
				if( first == AdrType::null ){
					first = slot;
				} else {
					arena[last].next = slot;
				}
				last = slot;
			}
		}
		if( last != AdrType::null ){
			arena[last].next = AdrType::null;
		}
		QueueType queue = QueueType(first,last);
		Index queue_index = random_uint(rand_state) % pool_size.adr;
		push_queue(queue,queue_index);
	}




	__device__ Index pull_span_atomic( Index* dst, Index count, Index stride, unsigned int& rand_state ){

		Index result = 0;
		for(unsigned int t=0; t<RETRY_COUNT; t++){
			Index queue_index = random_uint(rand_state) % pool_size.adr;
			QueueType queue = pull_queue(queue_index);
			if( ! queue.is_null() ){
				for( ; result<count; result++){
					Index old = atomicCAS(&dst[result*stride],AdrType::null,queue.get_head());
					if( old != AdrType::null ){
						continue;
					}
					AdrType adr = pop_front(queue);
					if( adr == AdrType::null ){
						break;
					}
				}
				push_queue(queue,queue_index);
			}
			if( result >= count ){
				return result;
			}
		}
		return result;
	}

	__device__ void push_span_atomic( Index* src, Index count, Index stride, unsigned int& rand_state ){
		Index first = AdrType::null;
		Index last  = AdrType::null;
		for( Index i=0; i<count; i++){
			Index swp = atomicExch(&src[stride*i],AdrType::null);
			if( swp != AdrType::null ){
				if( first == AdrType::null ){
					first = swp;
				} else {
					arena[last].next = swp;
				}
				last = swp;
			}
		}
		if( last != AdrType::null ){
			arena[last].next = AdrType::null;
		}
		QueueType queue = QueueType(first,last);
		Index queue_index = random_uint(rand_state) % pool_size.adr;
		push_queue(queue,queue_index);
	}


	#ifdef LAZY_MEM
	__device__ Index lazy_alloc_index(unsigned int& rand_state){

		#ifdef SMART_MEM
		__shared__ Index claim_index;
		unsigned int active = __activemask();
		__syncwarp(active);
		unsigned int count  = active_count();
		unsigned int offset = warp_inc_scan();
		if(current_leader()){
			if( claim_count.adr < arena_size.adr ){
				claim_index = atomicAdd(&(claim_count.adr),count);
			} else {
				claim_index = arena_size.adr;
			}
		}
		__syncwarp(active);
		if( (claim_index+offset) < arena_size.adr ){
			return claim_index + offset;
		}

		#else
		if( claim_count.adr < arena_size.adr ){
			Index claim_index = atomicAdd(&(claim_count.adr),1);
			if( claim_index < arena_size.adr ){
				return claim_index;
			}
		}
		#endif
		return AdrType::null;
	}
	#endif



	__device__ Index alloc_index(unsigned int& rand_state){


		#ifdef LAZY_MEM
		Index result = lazy_alloc_index(rand_state);
		if( result != AdrType::null ){
			return result;
		}
		#endif

		for( unsigned int t=0; t<RETRY_COUNT; t++){
			Index queue_index = random_uint(rand_state) % pool_size.adr;
			QueueType queue = pull_queue(queue_index);
			if( ! queue.is_null() ){
				AdrType adr = pop_front(queue);
				push_queue(queue,queue_index);
				return adr.adr;
			}
		}
		printf("{Alloc fail}");
		return AdrType::null;
	}


	__device__ DataType* alloc(unsigned int& rand_state){
		Index result_index = alloc_index(rand_state);
		if( result_index != Adr<Index>::null ){
			return &(arena[result_index].data);
		}
		return NULL;
	}


	__device__ void free(Index index, unsigned int& rand_state){
		if( index == AdrType::null ){
			return;
		}
		Index queue_index = random_uint(rand_state) % pool_size.adr;
		arena[index].next = AdrType::null;
		__threadfence();
		QueueType queue = QueueType(AdrType(index),AdrType(index));
		push_queue(queue,queue_index);
	}



	__device__ void free(DataType* address, unsigned int& rand_state){
		if( address == NULL ) {
			return;
		}
		Index index = address - arena;
		if( index == AdrType::null ){
			return;
		}
		free(index,rand_state);
	}



	__device__ void serial_init(){
		parallel_init(0,1);
	}



	__device__ void parallel_init(Index thread_index, Index thread_count){

		typedef MemPool<T,INDEX> PoolType;
		typedef INDEX Index;

		#ifdef LAZY_MEM
		claim_count = 0;

		for(Index i=thread_index; i<pool_size.adr; i+=thread_count){
			pool [i] = QueueType(AdrType::null,AdrType::null);
			//printf("{%d:(%d,%d)}",i,pool[i].get_head().adr,pool[i].get_tail().adr);
		}

		#else

		Index span = arena_size.adr / pool_size.adr;

		Index limit = arena_size.adr;
		for(Index i=thread_index; i<limit; i+=thread_count){
			if( ( (i%span) == (span-1) ) || ( i == (limit-1) ) ){
				arena[i].next = AdrType::null;
			} else {
				arena[i].next = i+1;
			}
			//printf("(%d:%d)",i,arena[i].next.adr);
		}

		for(Index i=thread_index; i<pool_size.adr; i+=thread_count){
			Index last;
			if( ((i+1)*span-1) >= arena_size.adr ) {
				last = arena_size.adr - 1;
			} else {
				last = (i+1)*span-1;
			}
			pool [i] = QueueType(AdrType(i*span),AdrType(last));
			//printf("{%d:(%d,%d)}",i,pool[i].get_head().adr,pool[i].get_tail().adr);
		}

		#endif
	}

};



template<typename T, typename INDEX>
__global__ void mempool_init(MemPool<T,INDEX> mempool){

	typedef INDEX Index;

	Index thread_count = blockDim.x * gridDim.x;
	Index thread_index = blockDim.x * blockIdx.x + threadIdx.x;

	mempool.parallel_init(thread_index,thread_count);
}




template<typename T, size_t SIZE>
struct MemCache {

	typedef typename T::DataType DataType;
	typedef typename T::Index    Index;
	typedef Adr<Index> AdrType;
	typedef MemPool<DataType,Index> PoolType;

	#ifdef LAZY_MEM
	bool try_lazy;
	#endif

	PoolType* parent;
	Index indexes[SIZE*WARP_SIZE];


	#if    __CUDA_ARCH__ < 600
		#define atomicExch_block atomicExch
		#define atomicCAS_block  atomicCAS
	#endif


	__device__ Index& get_index( unsigned int offset ){
		unsigned int real_offset = (WARP_SIZE*offset + threadIdx.x + (offset/SIZE)) % (SIZE*WARP_SIZE);
		return indexes[real_offset];
	}


	__device__ void initialize (PoolType& p) {
		if( current_leader() ){
			#ifdef LAZY_MEM
			try_lazy = true;
			#endif
			parent = &p;
		}
		for ( unsigned int i=0; i<SIZE; i++ ) {
			get_index(i) = AdrType::null;
		}
	}

	__device__ void finalize (unsigned int &rand_state) {
		for ( unsigned int i=0; i<SIZE; i++ ) {
			Index& slot = get_index(i);
			if( slot != AdrType::null ){
				parent->free(slot,rand_state);
			}
		}
	}



	__device__ Index alloc_index(unsigned int& rand_state){

		#if 0
		if( try_lazy ){
			Index result = parent->lazy_alloc_index(rand_state);
			if( result != AdrType::null ){
				return result;
			} else {
				try_lazy = false;
			}
		}
		#endif


		#if 0
		for ( unsigned int i=threadIdx.x; i<SIZE; i++ ) {
			Index swap = atomicExch_block(&(indexes[i]),AdrType::null);
			if( swap != AdrType::null ){
				return swap;
			}
		}
		for ( unsigned int i=0; i<threadIdx.x; i++ ) {
			Index swap = atomicExch_block(&(indexes[i]),AdrType::null);
			if( swap != AdrType::null ){
				return swap;
			}
		}
		#else
		for ( unsigned int i=0; i<(SIZE*WARP_SIZE); i++ ) {
			Index& slot = get_index(i);
			Index swap = atomicExch_block(&slot,AdrType::null);
			if( swap != AdrType::null ){
				return swap;
			}
		}
		#endif


		return parent->alloc_index(rand_state);
	}


	__device__ DataType* alloc(unsigned int& rand_state){
		Index result_index = alloc_index(rand_state);
		if ( result_index != Adr<Index>::null ) {
			return (parent->arena[result_index].data);
		}
		return NULL;
	}


	__device__ void free(Index index, unsigned int& rand_state){
		#if 0
		for ( unsigned int i=threadIdx.x; i<SIZE; i++ ) {
			Index swap = atomicCAS_block(&(indexes[i]),AdrType::null,index);
			if( swap == AdrType::null ){
				return;
			}
		}
		for ( unsigned int i=0; i<threadIdx.x; i++ ) {
			Index swap = atomicCAS_block(&(indexes[i]),AdrType::null,index);
			if( swap == AdrType::null ){
				return;
			}
		}
		#else
		for ( unsigned int i=0; i<(SIZE*WARP_SIZE); i++ ) {
			Index& slot = get_index(i);
			Index swap = atomicCAS_block(&slot,AdrType::null,index);
			if( swap == AdrType::null ){
				return;
			}
		}
		#endif
		parent->free(index,rand_state);
	}


	__device__ void free(DataType* address, unsigned int& rand_state){
		if( address == NULL ){
			return;
		}
		Index index = address - (parent->arena);
		free(index,rand_state);
	}

	#if    __CUDA_ARCH__ < 600
		#undef atomicExch_block
		#undef atomicCAS_block
	#endif


};




template<typename T, size_t SIZE>
struct SimpleMemCache {

	typedef typename T::DataType DataType;
	typedef typename T::Index    Index;
	typedef Adr<Index> AdrType;
	typedef MemPool<DataType,Index> PoolType;

	#ifdef LAZY_MEM
	bool try_lazy;
	#endif

	PoolType* parent;
	unsigned int counts[WARP_SIZE];
	Index indexes[SIZE*WARP_SIZE];


	#if    __CUDA_ARCH__ < 600
		#define atomicExch_block atomicExch
		#define atomicCAS_block  atomicCAS
	#endif


	__device__ Index& get_index( unsigned int offset ){
		unsigned int real_offset = (WARP_SIZE*offset + threadIdx.x + (offset/SIZE)) % (SIZE*WARP_SIZE);
		return indexes[real_offset];
	}


	__device__ void initialize (PoolType& p) {
		if( current_leader() ){
			parent = &p;
		}
		counts[threadIdx.x] = 0;
	}

	__device__ void finalize (unsigned int &rand_state) {
		unsigned int count = counts[threadIdx.x];
		for ( unsigned int i=0; i<count; i++ ) {
			Index& slot = get_index(i);
			parent->free(slot,rand_state);
		}
	}


	__device__ void fill_up(unsigned int& rand_state) {
		unsigned int count = counts[threadIdx.x];
		if( SIZE/2 > count ) {
			parent->pull_span( &get_index(count), (SIZE/2)-count, WARP_SIZE, rand_state);
		}
	}


	__device__ Index alloc_index(unsigned int& rand_state) {

		unsigned int& count = counts[threadIdx.x];
		if( count > 0 ){
			count -= 1;
			Index& slot = get_index(count);
			return slot;
		} else {
			return parent->alloc_index(rand_state);
		}
	}


	__device__ DataType* alloc(unsigned int& rand_state){
		Index result_index = alloc_index(rand_state);
		if ( result_index != Adr<Index>::null ) {
			return (parent->arena[result_index].data);
		}
		return NULL;
	}


	__device__ void free(Index index, unsigned int& rand_state){

		unsigned int& count = counts[threadIdx.x];
		if( count < SIZE ){
			get_index(count) = index;
			count += 1;
		} else {
			parent->free(index,rand_state);
		}

	}


	__device__ void free(DataType* address, unsigned int& rand_state){
		if( address == NULL ){
			return;
		}
		Index index = address - (parent->arena);
		free(index,rand_state);
	}

	#if    __CUDA_ARCH__ < 600
		#undef atomicExch_block
		#undef atomicCAS_block
	#endif


};






template<typename T, typename INDEX, typename MASK = unsigned int>
struct MemChunk {

	typedef INDEX Index;
	typedef MASK  Mask;
	typedef MemPool<MemChunk<T,Index,Mask>,Index> PoolType;

	static const Index SIZE = std::numeric_limits<Mask>::digits;
	static const Mask  FULL = std::numeric_limits<Mask>::max();
	static const MASK  HIGH_BIT = 1<<(SIZE-1);
	static const Index OFFSET_MASK = ~(SIZE-1);

	Mask fill_mask;
	T data[SIZE];

	__device__ Index offset(PoolType& parent){
		return (this - parent.arena);
	}

	__device__ bool full(){
		return (fill_mask == FULL);
	}

	__device__ bool empty(){
		return (fill_mask == 0);
	}

	__device__ bool fill_count(){
		return __popc(fill_mask);
	}

	__device__ MemChunk<T,Index,Mask>()
		: fill_mask(0)
	{}

	__device__ Index alloc(PoolType& parent, Mask& rand_state) {
		Mask try_mask = random_uint(rand_state);
		while( ! full() ){
			Mask scan_mask = try_mask & ~fill_mask;
			scan_mask = (scan_mask == 0) ? ~fill_mask : scan_mask;
			Index lead = leading_zeros(scan_mask);
			Mask select = HIGH_BIT >> lead;
			if( select == 0 ){
				break;
			}
			Mask prior = atomicOr(fill_mask,select);
			if( (prior & select) == 0 ){
				return ( SIZE * offset(parent) ) + ( SIZE - 1 - lead );
			}
		}
		return Adr<Index>::null;
	}

	__device__ void free(Index index, PoolType& parent, Mask& rand_state) {
		if( (index & OFFSET_MASK) == offset() ){
			Index inner_index = index & ~OFFSET_MASK;
			Mask  select = 1 << inner_index;
			atomicAnd(fill_mask, ~select);
		}
	}

};



template<typename T, typename INDEX, typename MASK, INDEX SIZE_VAL>
struct MemPoolBank {

	typedef INDEX Index;
	typedef MASK  MaskType;
	typedef MemChunk<T,Index,MaskType> ChunkType;
	typedef MemPool<ChunkType,Index>   PoolType;

	static const Index SIZE = SIZE_VAL;

	unsigned int rand_state;
	Index full_count;

	PoolType& parent;
	Index chunks[SIZE];

	__device__ Index containing_chunk(Index target){
		Index result = ChunkType::offset_mask & index;
		if( result >= parent.arena_size ){
			return Adr<Index>::null;
		}
		return result;
	}

	__device__ MemPoolBank<T,Index,MaskType,SIZE_VAL>
	(PoolType& parent_ref, unsigned int seed, bool& err)
		: parent(parent_ref)
		, full_count(0)
		, rand_state(seed)
	{
		for(int i=0; i<SIZE; i++){
			chunks[i] = Adr<Index>::null;
		}
		err = parent.pull_span(chunks,SIZE,seed);
	}


	__device__ Index alloc() {
		for(int i=0; i<SIZE; i++){
			if( chunks[i] == Adr<Index>::null ){
				continue;
			}

		}
	}

	__device__ Index hard_neighbor_alloc(Index target, MaskType& rand_state){
		return parent.arena[ChunkType::offset_mask & index].alloc(parent,rand_state);
	}

	__device__ Index neighbor_alloc(Index target, MaskType& rand_state){
		Index result = hard_neighbor_alloc(target,rand_state);
		if( result == Adr<Index>::null){
			return alloc(rand_state);
		} else {
			return result;
		}
	}


	__device__ Index coalesce(Index original) {

	}

	__device__ void free(Index index) {
		Index chunk_index = ChunkType::offset_mask & index;
		if( index < parent.arena_size ){
			parent.arena[chunk_index].free(parent,index);
		}
	}

};




}



