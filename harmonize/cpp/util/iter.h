

#ifndef HARMONIZE_UTIL_ITER
#define HARMONIZE_UTIL_ITER

namespace util {

namespace iter {

template<typename INDEX_TYPE> struct GroupIter;
template<typename INDEX_TYPE> struct AtomicIter;

template<typename INDEX_TYPE>
struct Iter
{

	typedef INDEX_TYPE IndexType;

	IndexType value;
	IndexType limit;

	__host__ __device__ Iter<IndexType> (IndexType v, IndexType l)
	: value(v), limit(l) {}

	__host__ __device__ Iter<IndexType> ( Iter<IndexType> const &iter )
		: value(iter.value)
		, limit(iter.limit)
	{}

	__host__ __device__ Iter<IndexType> () = default;

	__device__ void reset(IndexType start_val, IndexType limit_val) {
		value = start_val;
		limit = limit_val;
	}

	__host__ __device__ bool step(IndexType& iter_val){
		if( value >= limit ){
			return false;
		}

		iter_val = value;
		value   += 1;

		return true;
	}

	__host__ __device__ Iter<IndexType> leap(IndexType length) {
		IndexType sub_limit = value + length;
		sub_limit = (sub_limit > limit) ? limit : sub_limit;
		IndexType old_value = value;
		value = sub_limit;
		return Iter<IndexType>(old_value,sub_limit);
	}

	__host__ __device__ bool done() const {
		return (value >= limit);
	}

};



template<typename INDEX_TYPE>
struct GroupIter
{


	typedef INDEX_TYPE IndexType;

	IndexType value;
	IndexType limit;

	__host__ __device__ GroupIter<IndexType> ( Iter<IndexType> iter )
		: value(iter.value)
		, limit(iter.limit)
	{}

	__host__ __device__ GroupIter<IndexType> () = default;

	__device__ void reset(IndexType start_val, IndexType limit_val) {
		__syncthreads();
		if( current_leader() ){
			value = start_val;
			limit = limit_val;
		}
		__syncthreads();
	}


	__device__ bool step(IndexType& iter_val) {

		if( value < limit ){
			IndexType val = value + threadIdx.x;
			if ( val < limit ){
				iter_val = val;
			}
			__syncthreads();
			if( current_leader() ){
				value += blockDim.x;
			}
			__syncthreads();
			return val < limit;
		}
		return false;

	}


	__device__ Iter<IndexType> leap(IndexType length) {

		Iter<IndexType> result(0,0,blockDim.x);
		if( value < limit ){
			IndexType start_val = value + threadIdx.x;
			IndexType limit_val = start_val + blockDim.x * length;
			result.value = (start_val < limit) ? start_val : limit;
			result.limit = (limit_val < limit) ? limit_val : limit;
			__syncthreads();
			if( current_leader() ){
				value += blockDim.x * length;
			}
			__syncthreads();
		}
		return result;

	}


	__host__ __device__ bool done() const {
		return ( value >= limit );
	}

};




template<typename INDEX_TYPE>
struct AtomicIter
{

	typedef INDEX_TYPE IndexType;

	IndexType value;
	IndexType limit;


	__host__ __device__ AtomicIter<IndexType> () = default;

	__host__ __device__ AtomicIter<IndexType> ( IndexType start_val, IndexType limit_val )
		: value(start_val)
		, limit(limit_val)
	{}

	__host__ __device__ AtomicIter<IndexType> ( Iter<IndexType> iter )
		: value(iter.value)
		, limit(iter.limit)
	{}

	__host__ __device__ AtomicIter<IndexType>& operator= ( Iter<IndexType> iter )
	{
		value = iter.value;
		limit = iter.limit;
		return *this;
	}

	__device__ void reset(IndexType start_val, IndexType limit_val) {
		__threadfence();
		IndexType old_limit = atomicAdd(&limit,0);
		if( old_limit > limit_val ){
			atomicExch(&limit,limit_val);
		} else if ( old_limit < limit_val ) {
			atomicExch(&value,limit_val);
			atomicExch(&limit,limit_val);
		}
		atomicExch(&value,start_val);
		__threadfence();
	}


	__device__ Iter<IndexType> leap(IndexType leap_size) {

		IndexType start_val = 0;
		IndexType limit_val = 0;

		if( value < limit ){
			__threadfence();
			start_val = atomicAdd(&value,leap_size);
			__threadfence();
			limit_val = start_val + leap_size;
			start_val = (start_val < limit) ? start_val : limit;
			limit_val = (limit_val < limit) ? limit_val : limit;
		}

		return Iter<IndexType>(start_val,limit_val);
	}


	__device__ bool step(IndexType& iter_val) {

		if( value >= limit ){
			return false;
		}

		__threadfence();
		IndexType try_val = atomicAdd(&value,1);
		__threadfence();

		if( try_val >= limit ){
			return false;
		}

		iter_val = try_val;

		return true;
	}

	__device__ bool done() const {
		__threadfence();
		bool result = ( value >= limit );
		__threadfence();
		return result;
	}

};




template< typename T, template < typename > typename ITER_TYPE = Iter, typename INDEX_TYPE = unsigned int >
struct ArrayIter {

	typedef INDEX_TYPE IndexType;
	typedef ITER_TYPE<IndexType> IterType;

	T* array;
	IterType iter;

	template<template < typename > class OTHER_ITER_TYPE>
	__host__ __device__ ArrayIter<T,ITER_TYPE,INDEX_TYPE> (ArrayIter<T,OTHER_ITER_TYPE,INDEX_TYPE>&& other_iter)
		: array(other_iter.array)
		, iter(other_iter.iter)
	{}

	template<template < typename > class OTHER_ITER_TYPE>
	__host__ __device__ ArrayIter<T,ITER_TYPE,INDEX_TYPE>& operator= (ArrayIter<T,OTHER_ITER_TYPE,INDEX_TYPE>&& other_iter)
	{
		array = other_iter.array;
		iter  = other_iter.iter;
		return *this;
	}


	__host__ __device__ ArrayIter<T,ITER_TYPE,INDEX_TYPE> (T* adr, IterType other_iter)
		: array(adr)
		, iter (other_iter)
	{}

	__host__ __device__ ArrayIter<T,ITER_TYPE,INDEX_TYPE> () = default;


	__device__ void reset(T* new_array, IterType new_iter) {
		array = new_array;
		iter  = new_iter;
	}


	__device__ bool step_val(T  &val){
		IndexType index;
		if( iter.step(index) ){
			val = array[index];
			return true;
		}
		return false;
	}


	__device__ bool step_idx_val(IndexType& idx, T  &val){
		IndexType index;
		if( iter.step(index) ){
			idx = index;
			val = array[index];
			return true;
		}
		return false;
	}


	__device__ bool step_ptr(T *&val){
		IndexType index;
		if( iter.step(index) ){
			val = &(array[index]);
			return true;
		}
		return false;
	}

	__device__ bool step_idx_ptr(IndexType& idx, T *&val){
		IndexType index;
		if( iter.step(index) ){
			idx = index;
			val = &(array[index]);
			return true;
		}
		return false;
	}

	__device__ ArrayIter<T,Iter,IndexType> leap(IndexType leap_size) {
		return ArrayIter<T,Iter,IndexType>(array,iter.leap(leap_size));
	}

	__device__ bool done() {
		return iter.done();
	}

};





template<typename T, typename INDEX_TYPE = unsigned int>
struct IOBuffer
{

	typedef INDEX_TYPE IndexType;

	bool  toggle; //True means A is in and B is out. False indicates vice-versa.
	T    *data_a;
	T    *data_b;

	IndexType capacity;

	AtomicIter<IndexType> input_iter;
	AtomicIter<IndexType> output_iter;


	__host__ __device__  IOBuffer<T,IndexType>()
		: capacity(0)
		, toggle(false)
		, input_iter (0,0)
		, output_iter(0,0)
		, data_a(NULL)
		, data_b(NULL)
	{}

	__device__  IOBuffer<T,IndexType>(IndexType cap,T* a, T* b)
		: capacity(cap)
		, toggle(false)
		, input_iter (0,0  )
		, output_iter(0,cap)
		, data_a(a)
		, data_b(b)
	{}

	__host__  IOBuffer<T,IndexType>(IndexType cap)
		: capacity(cap)
		, toggle(false)
		, input_iter (0,0  )
		, output_iter(0,cap)
	{}

	__host__ void host_init()
	{
		data_a = host::hardMalloc<T>( capacity );
		data_b = host::hardMalloc<T>( capacity );
	}

	__host__ void host_free()
	{
		if ( data_a != NULL ) {
			host::auto_throw( adapt::GPUrtFree( data_a ) );
		}

		if ( data_b != NULL ) {
			host::auto_throw( adapt::GPUrtFree( data_b ) );
		}
	}


	__host__ __device__ T* input_pointer(){
		return toggle ? data_b : data_a;
	}

	__host__ __device__ T* output_pointer(){
		return toggle ? data_a : data_b;
	}

	__device__ ArrayIter<T,Iter,IndexType> pull_span(IndexType pull_size)
	{
		Iter<IndexType> pull_iter = input_iter.leap(pull_size);
		return ArrayIter<T,Iter,IndexType>(input_pointer(),pull_iter);
	}

	__device__ ArrayIter<T,Iter,IndexType> push_span(IndexType push_size)
	{
		Iter<IndexType> push_iter = output_iter.leap(push_size);
		return ArrayIter<T,Iter>(output_pointer(),push_iter);
	}

	__device__ bool pull(T& value){
		IndexType index;
		if( ! input_iter.step(index) ){
			return false;
		}
		value = input_pointer()[index];
		return true;
	}

	__device__ bool pull_index(IndexType& index){
		return input_iter.step(index);
	}

	__device__ bool push(T value){
		IndexType index;
		if( ! input_iter.step(index) ){
			return false;
		}
		input_pointer()[index] = value;
		return true;
	}



	__device__ bool push_index(IndexType& index){
		return output_iter.step(index);
	}

	__device__ void flip()
	{
		toggle = !toggle;
		IndexType in_count = output_iter.value >= capacity ? capacity : output_iter.value;
		//printf("{Flipped with output at %d.}",in_count);
		input_iter  = AtomicIter<IndexType>(0,in_count);
		output_iter = AtomicIter<IndexType>(0,capacity);
	}

	__device__ bool input_empty()
	{
		return input_iter.done();
	}

	__device__ bool output_full()
	{
		return output_iter.done();
	}

	__device__ float output_fill_fraction(){
		return ((float) atomicAdd(&(output_iter.value),0u)) / ((float) capacity);
	}


	__device__ float output_fill_fraction_sync(){

		__shared__ INDEX_TYPE progress;

		__syncthreads();
		if( threadIdx.x == 0 ){
			progress = atomicAdd(&(output_iter.value),0u);
		}
		__syncthreads();

		return ((float) progress) / ((float) capacity);
	}
};




#if 0
template<typename T> bool tie_breaker (T& A, T& B);


// Experimental population control mechanism
template<typename T, typename INDEX_TYPE = unsigned int, typename HASH_TYPE = unsigned int>
struct MCPCBuffer {

	typedef INDEX_TYPE IndexType;
	typedef HASH_TYPE HashType;

	struct LinkType {
		HashType hash;
		IndexType next;
		T        data;
	};

	struct LinkJump {
		HashType hash;
		IndexType index;
	};

	IndexType capacity;
	IndexType overflow;


	IOBuffer<LinkType,IndexType> link_buffer;

	IOBuffer<LinkJump,IndexType> jump_buffer;


	__device__ bool pull(T& dest){
		while ( ! jump_buffer.input_empty() ) {
			IndexType idx;
			if ( jump_buffer.pull_idx(idx) ){
				LinkJump& jump = jump_buffer.input_ptr[idx];
				if( jump.next == Adr<IndexType>::null ){
					continue;
				}
				T best = link_buffer[];
				while () {

				}
				jump.hash  = 0;
				jump.index = Adr<IndexType>::null;
			}
		}
		return false;
	}

	__device__ void push(T value, IndexType index, HashType hash){
		HashType atom_max = atomicMax(&(LinkJump[index].hash),hash);
		if ( atom_max <= hash ) {

		} else if ( atom_max == hash ) {

		}


	}

	__device__ void flip(){
		link_buffer.flip();

	}

	__device__ bool input_empty() {
		return link_buffer.input_empty();
	}

	__device__ bool output_full() {
		return link_buffer.output_full();
	}

};
#endif


}
}




#endif


