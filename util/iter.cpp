






template<typename ITER_TYPE>
struct Iter
{

	typedef ITER_TYPE IterType;

	IterType value;
	IterType limit;
	IterType tempo;


	__host__ __device__ Iter<ITER_TYPE> (IterType v, IterType l, IterType w)
	: value(v), limit(l), tempo(w) {}


	 __device__ bool step(IterType& iter_val){

		if( value >= limit ){
			return false;
		}

		iter_val = value;
		value   += tempo;

		return true;
	}

	__device__ bool done() const {
		return (value >= limit);
	}


};



template<typename ITER_TYPE>
struct GroupIter
{


	typedef ITER_TYPE IterType;

	IterType value;
	IterType limit;

	 __device__ void reset(IterType start_val, IterType limit_val) {
		__syncthreads();
		if( current_leader() ){
			value = start_val;
			limit = limit_val;
		}
		__syncthreads();
	}


	 __device__ bool step(IterType& iter_val) {
		
		if( value < limit ){
			IterType val = value + threadIdx.x;	
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


	 __device__ Iter<IterType> leap(IterType length) {
	
		Iter<IterType> result(0,0,blockDim.x);
		if( value < limit ){
			IterType start_val = value + threadIdx.x;	
			IterType limit_val = start_val + blockDim.x * length;
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


	 __device__ bool done() const {
		
		return ( value >= limit );
		
	}

};




template<typename ITER_TYPE>
struct AtomicIter
{

	typedef ITER_TYPE IterType;

	IterType value;
	IterType limit;

	 __device__ void reset(IterType start_val, IterType limit_val) {
		value = start_val;
		limit = limit_val;
	}


	 __device__ GroupIter<IterType> group_leap(IterType leap_size) {
	
		GroupIter<IterType> result;
		result.value = limit;
		result.limit = limit;
	
		if( value <= limit ){
			IterType start_val = atomicAdd(&value,leap_size);	
			IterType limit_val = start_val + leap_size;
			result.value = (start_val < limit) ? start_val : limit;
			result.limit = (limit_val < limit) ? limit_val : limit;
		}
		return result;
		
	}

	 __device__ Iter<IterType> leap(IterType leap_size) {
	
		Iter<IterType> result(0,0,0);

		__shared__ IterType start_val;
		__shared__ IterType limit_val;

		if( current_leader() ){
			start_val = 0;
			limit_val = 0;
			if( value < limit ){
				start_val = atomicAdd(&value,leap_size*blockDim.x);
				limit_val = start_val + leap_size*blockDim.x;
				start_val = (start_val < limit) ? start_val : limit;
				limit_val = (limit_val < limit) ? limit_val : limit;
			}
		}
		__syncthreads();

		result.value = start_val + threadIdx.x;
		result.limit = limit_val;
		result.tempo = blockDim.x;

		return result;
		
	}


	 __device__ bool step(IterType& iter_val) {
	
		if( value >= limit ){
			return false;
		}

		IterType try_val = atomicAdd(&value,1);
		
		if( try_val >= limit ){
			return false;
		}

		iter_val = try_val;

		return true;
		
	}


	 __device__ bool done() const {
		
		return ( value >= limit );
		
	}


	__host__ __device__ AtomicIter<IterType> () {}

	__host__ __device__ AtomicIter<IterType> ( IterType start_val, IterType limit_val )
		: value(start_val)
		, limit(limit_val)
	{}

};





template<typename ITER_TYPE>
__device__ Iter<ITER_TYPE> tiered_leap (
	AtomicIter<ITER_TYPE> &glb, ITER_TYPE device_leap,
	GroupIter  <ITER_TYPE> &wrp, ITER_TYPE group_leap
) {

	if( ! wrp.done() ){
		return wrp.leap(group_leap);
	}

	if( !glb.done() ){
		if( current_leader() ){
			wrp = glb.group_leap(device_leap);
		}
		__syncthreads();
		return wrp.leap(group_leap);
	}

	return Iter<ITER_TYPE>(0,0,0);

}


template<typename T,typename ITER_TYPE = unsigned int>
struct ArrayIter {

	typedef ITER_TYPE IterType;

	T* array;
	Iter<IterType> iter;

	__device__ bool step_val(T  &val){
		IterType index;
		if( iter.step(index) ){
			val = array[index];
			return true;
		}
		return false;
	}


	__device__ bool step_idx_val(IterType& idx, T  &val){
		IterType index;
		if( iter.step(index) ){
			idx = index;
			val = array[index];
			return true;
		}
		return false;
	}


	__device__ bool step_ptr(T *&val){
		IterType index;
		if( iter.step(index) ){
			val = &(array[index]);
			return true;
		}
		return false;
	}
	
	__device__ bool step_idx_ptr(IterType& idx, T *&val){
		IterType index;
		if( iter.step(index) ){
			idx = index;
			val = &(array[index]);
			return true;
		}
		return false;
	}

	__device__ ArrayIter<T,IterType> (T* adr, Iter<IterType> itr)
		: array(adr)
		, iter (itr)
	{}
	
	__device__ ArrayIter<T,IterType> ()
		: array(NULL)
		, iter(0,0,0)
	{}

};






template<typename T,typename ITER_TYPE = unsigned int>
struct GroupArrayIter {

	typedef ITER_TYPE IterType;

	T* array;
	GroupIter<IterType> iter;

	__device__ bool step_val(T  &val){
		IterType index;
		if( iter.step(index) ){
			val = array[index];
			return true;
		}
		return false;
	}


	__device__ bool step_idx_val(IterType& idx, T  &val){
		IterType index;
		if( iter.step(index) ){
			idx = index;
			val = array[index];
			return true;
		}
		return false;
	}


	__device__ bool step_ptr(T *&val){
		IterType index;
		if( iter.step(index) ){
			val = &(array[index]);
			return true;
		}
		return false;
	}
	
	__device__ bool step_idx_ptr(IterType& idx, T *&val){
		IterType index;
		if( iter.step(index) ){
			idx = index;
			val = &(array[index]);
			return true;
		}
		return false;
	}

	__device__ GroupArrayIter<T,IterType> (T* adr, GroupIter<IterType> itr)
		: array(adr)
		, iter (itr)
	{}

	GroupArrayIter<T,IterType> () = default;

	__device__ ArrayIter<T,IterType> leap(IterType length) {
		return ArrayIter<T,IterType>(array,iter.leap(length));
	}
	
};



template<typename T, typename ITER_TYPE = unsigned int>
struct IOBuffer
{

	typedef ITER_TYPE IterType;

	bool  toggle; //True means A is in and B is out. False indicates vice-versa.
	T    *data_a;
	T    *data_b;

	IterType capacity;

	AtomicIter<IterType> input_iter;
	AtomicIter<IterType> output_iter;


	__device__  IOBuffer<T,IterType>()
		: capacity(0)
		, toggle(false)
		, input_iter (0,0)
		, output_iter(0,0)
		, data_a(NULL)
		, data_b(NULL)
	{}

	__device__  IOBuffer<T,IterType>(IterType cap,T* a, T* b)
		: capacity(cap)
		, toggle(false)
		, input_iter (0,0  )
		, output_iter(0,cap)
		, data_a(a)
		, data_b(b)
	{}

	__host__  IOBuffer<T,IterType>(IterType cap)
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
			host::auto_throw( cudaFree( data_a ) );
		}

		if ( data_b != NULL ) {
			host::auto_throw( cudaFree( data_b ) );
		}
	}


	__device__ T* input_ptr(){
		return toggle ? data_b : data_a;
	}

	__device__ T* output_ptr(){
		return toggle ? data_a : data_b;
	}

	__device__ ArrayIter<T,IterType> pull_span(IterType pull_size)
	{
		Iter<IterType> pull_iter = input_iter.leap(pull_size);
		return ArrayIter<T,IterType>(input_ptr(),pull_iter);
	}

	__device__ ArrayIter<T,IterType> push_span(IterType push_size)
	{
		Iter<IterType> push_iter = output_iter.leap(push_size);
		return ArrayIter<T,IterType>(output_ptr(),push_iter);
	}


	__device__ GroupArrayIter<T,IterType> pull_group_span(IterType pull_size)
	{
		GroupIter<IterType> pull_iter = input_iter.group_leap(pull_size);
		return GroupArrayIter<T,IterType>(input_ptr(),pull_iter);
	}

	__device__ GroupArrayIter<T,IterType> push_group_span(IterType push_size)
	{
		GroupIter<IterType> push_iter = output_iter.group_leap(push_size);
		return GroupArrayIter<T,IterType>(output_ptr(),push_iter);
	}

	__device__ bool pull(T& value){
		IterType index;
		if( ! input_iter.step(index) ){
			return false;
		}
		value = input_ptr()[index];
		return true;
	}

	__device__ bool pull_idx(IterType& index){
		return input_iter.step(index);
	}

	__device__ bool push(T value){
		IterType index;
		if( ! input_iter.step(index) ){
			return false;
		}
		input_ptr()[index] = value;
		return true;
	}



	__device__ bool push_idx(IterType& index){
		return output_iter.step(index);
	}

	__device__ void flip()
	{
		toggle = !toggle;	
		input_iter  = AtomicIter<IterType>(0,output_iter.value);
		output_iter = AtomicIter<IterType>(0,capacity);
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

		__shared__ ITER_TYPE progress;

		__syncthreads();
		if( threadIdx.x == 0 ){
			progress = atomicAdd(&(output_iter.value),0u);
		}
		__syncthreads();

		return ((float) progress) / ((float) capacity); 
	}
};














