#pragma once

#include "math.h"
#include <vector>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <limits>
#include <memory>



namespace util {



template<typename T>
class Option {

	T    data;
	bool some;

	operator bool() { return some; }

	T unwrap()
	{
		if ( some ) {
			return data;
		}
		throw std::string("Attempted to unwrap empty Option.");
	}

};



template<typename OKAY, typename FAIL>
class Result {

	union OkayOrFail {
		OKAY okay;
		FAIL fail;
	};

	OkayOrFail data;
	bool       okay;


	operator bool() { return okay; }

	Result<OKAY,FAIL>(OKAY value){
		data.okay = value;
		okay = true;
	}

	Result<OKAY,FAIL>(FAIL value){
		data.fail = value;
		okay = false;
	}

	static Result<OKAY,FAIL> wrap(OKAY value){
		Result<OKAY,FAIL> result;
		result.data.okay = value;
		result.okay = true;
		return result;
	}
	
	static Result<OKAY,FAIL> wrap_fail(FAIL value){
		Result<OKAY,FAIL> result;
		result.data.fail = value;
		result.okay = false;
		return result;
	}

	OKAY unwrap()
	{
		if ( okay ) {
			return data.okay;
		}
		throw std::string("Attempted to unwrap failed Result.");
	}

	FAIL unwrap_fail()
	{
		if ( !okay ) {
			return data.fail;
		}
		throw std::string("Attempted to unwrap failed Result.");
	}


	OKAY unwrap_or(OKAY other){
		return okay ? data.okay : other;
	}
	
	FAIL unwrap_fail_or(FAIL other){
		return !okay ? data.fail : other;
	}

	
	template<typename FUNCTION>
	auto and_then(FUNCTION f) -> Result<decltype(f(data.okay)),FAIL>
	{
		typedef Result<decltype(f(data.okay)),FAIL> RetType;

		if( ! okay ){
			return RetType::wrap_fail(data.fail);
		} else {
			return RetType::wrap_okay(f(data.okay));
		}
	}



};


void auto_throw(cudaError_t value){
	if ( value != cudaSuccess ) { throw value; }
}


template<typename T>
T* hardMalloc(size_t size){
	T* result;
	auto_throw( cudaMalloc(&result, sizeof(T)*size) );
	return result;
}



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
	static const PairType LEFT_MASK  = RIGHT_MASK << HALF_WIDTH;

	PairType data;

	__host__  __device__ T    get_left() {
		return (data >> HALF_WIDTH) & RIGHT_MASK;
	}

	__host__  __device__ void set_left(T val) {
		data &= RIGHT_MASK;
		data |= ((PairType) val) << HALF_WIDTH;
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





template<typename T>
class DevBuf {

	protected:

	struct Inner {

		size_t  size;
		T       *adr;
		
		Inner(T *adr_val, size_t size_val)
		: adr (adr_val ) 
		, size(size_val)
		{}

		~Inner() {
			if ( adr != NULL) {
				cudaFree(adr);
			}
		}

	};

	std::shared_ptr<Inner> inner;

	public:


	operator T*&() { return inner->adr; }

	__host__ void resize(size_t s){
		T* new_adr = hardMalloc<T>(s);
		size_t copy_count = ( s < inner->size ) ? s : inner->size;
		auto_throw( cudaMemcpy(
			new_adr,
			inner->adr,
			sizeof(T)*copy_count,
			cudaMemcpyDeviceToDevice
		) );
		auto_throw( cudaFree(inner->adr) );
		inner->size = s;
		inner->adr = new_adr;
	}

	__host__ void operator<<(std::vector<T> &other) {
		if( other.size() != inner->size ){
			resize(other->size);
		}
		auto_throw( cudaMemcpy(
			inner->adr,
			other.data(),
			sizeof(T)*inner->size,
			cudaMemcpyHostToDevice
		) );
	}

	__host__ void operator>>(std::vector<T> &other) {
		if( other.size() != inner->size ){
			other.resize(inner->size);
		}
		auto_throw( cudaMemcpy(
			other.data(),
			inner->adr,
			sizeof(T)*inner->size,
			cudaMemcpyDeviceToHost
		) );
	}


	__host__ void operator<<(T &other) {
		if( inner->size != 1 ){
			resize(1);
		}
		auto_throw( cudaMemcpy(
			inner->adr,
			&other,
			sizeof(T),
			cudaMemcpyHostToDevice
		) );
	}

	__host__ void operator<<(T &&other) {
		T host_copy = other;
		if( inner->size != 1 ){
			resize(1);
		}
		auto_throw( cudaMemcpy(
			inner->adr,
			&host_copy,
			sizeof(T),
			cudaMemcpyHostToDevice
		) );
	}



	DevBuf<T> (T* adr, size_t size)
		: inner(new Inner(adr,size))
	{}

	DevBuf<T> ()
		: DevBuf<T>((T*)NULL,(size_t)0)
	{}

	DevBuf<T> (size_t size)
		: DevBuf<T> (hardMalloc<T>(size),size)
	{}
	
	DevBuf<T> (T& value)
		: DevBuf<T>()
	{
		(*this) << value;
	}

	DevBuf<T> (T&& value)
		: DevBuf<T>()
	{
		(*this) << value;
	}

	
	template<typename... ARGS>
	static DevBuf<T> make(ARGS... args)
	{
		return DevBuf<T>( T(args...) );
	}
	

};




template<typename T>
class DevObj {

	protected:

	struct Inner {

		T *adr;
		T host_copy;


		void push_data(){
			printf("Pushing data into %p\n",adr);
			auto_throw( cudaMemcpy(
				adr,
				&host_copy,
				sizeof(T),
				cudaMemcpyHostToDevice
			) );
		}

		void pull_data(){
			printf("Pulling data from %p\n",adr);
			auto_throw( cudaMemcpy(
				&host_copy,
				adr,
				sizeof(T),
				cudaMemcpyDefault
			) );
		}
		
		template<typename... ARGS>
		Inner(T *adr_val, ARGS... args)
			: adr (adr_val)
			, host_copy(args...)
		{
			host_copy.host_init();
			push_data();
		}

		~Inner() {
			if ( adr != NULL) {
				printf("Doing a free\n");
				pull_data();
				host_copy.host_free();
				cudaFree(adr);
			}
		}

	};

	std::shared_ptr<Inner> inner;

	public:


	void push_data(){
		inner->push_data();
	}

	void pull_data(){
		inner->pull_data();
	}

	operator T*() { return inner->adr; }


	template<typename... ARGS>
	DevObj<T>(ARGS... args)
		: inner(new Inner(hardMalloc<T>(1),args...))
	{}
	
};





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
// A simple pseudo-random number generator. This algorithm should never be used for cryptography, 
// it is simply used to generate numbers random enough to reduce collisions for atomic
// instructions performed to manage the runtime state.
*/
 __device__ unsigned int random_uint(unsigned int &state){

	state = (0x10DCDu * state + 1u);
	return state;

}



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
struct WarpIter
{


	typedef ITER_TYPE IterType;

	IterType value;
	IterType limit;

	 __device__ void reset(IterType start_val, IterType limit_val) {
		__syncwarp();
		if( current_leader() ){
			value = start_val;
			limit = limit_val;
		}
		__syncwarp();
	}


	 __device__ bool step(IterType& iter_val) {
		
		if( value < limit ){
			IterType val = value + threadIdx.x;	
			if ( val < limit ){
				iter_val = val;
			}
			__syncwarp();
			if( current_leader() ){
				val += blockDim.x;
			}
			__syncwarp();
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
			__syncwarp();
			if( current_leader() ){
				value += blockDim.x * length;
			}
			__syncwarp();
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


	 __device__ WarpIter<IterType> warp_leap(IterType leap_size) {
	
		WarpIter<IterType> result;
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
		__syncwarp();

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
	AtomicIter<ITER_TYPE> &glb, ITER_TYPE global_leap,
	WarpIter  <ITER_TYPE> &wrp, ITER_TYPE warp_leap
) {

	if( ! wrp.done() ){
		return wrp.leap(warp_leap);
	}

	if( !glb.done() ){
		if( current_leader() ){
			wrp = glb.warp_leap(global_leap);
		}
		__syncwarp();
		return wrp.leap(warp_leap);
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
		data_a = hardMalloc<T>( capacity );
		data_b = hardMalloc<T>( capacity );
	}

	__host__ void host_free()
	{
		auto_throw( cudaFree( data_a ) );
		auto_throw( cudaFree( data_b ) );
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


	__device__ bool pull(T& value){
		IterType index;
		if( ! input_iter.step(index) ){
			return false;
		}
		value = input_ptr()[index];
		return true;
	}

	__device__ bool pull_idx(IterType& value){
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

};



template<typename T, typename INDEX>
struct MemPool {

	typedef INDEX Index;

	static const Index null = 0;

	T*     arena;
	Index  arena_size;

	Index* pool;
	Index  pool_size;


	__host__ void host_init()
	{
		arena  = hardMalloc<T>    ( arena_size );
		pool   = hardMalloc<Index>( pool_size  );
	}

	__host__ void host_free()
	{
		auto_throw( cudaFree( arena ) );
		auto_throw( cudaFree( pool  ) );
	}


	

	__device__ Index pull( Index count, unsigned int& rand_state ){
		
	}

	__device__ void  push( Index count, unsigned int& rand_state ){

	}

	__device__ T& operator[] (Index index){
		return arena[index];
	}

	__device__ Index& next(Index index){
		return *((Index*)&(arena[index]));
	}


};



template<typename T, typename INDEX>
__global__ void mempool_init(MemPool<T,INDEX>& mempool){

	typedef MemPool<T,INDEX> PoolType;
	typedef INDEX Index;

	int limit = mempool.arena_size - 1;
	for(Index i=0; i<limit; i++){
		mempool.next(i) = i+1;
	}

	Index span = mempool.arena_size / mempool.pool_size;
	for(Index i=0; i<mempool.pool_size; i++){
		mempool.pool [i] = i*span;
		if( i != 0 ){
			mempool.next(i*span-1) = PoolType::null;
		}
	}

}




// Experimental population control mechanism
#if 0
template<typename T, typename ID>
struct TitanicValue {

	PairPack<ID>

};




template<typename T, typename ITER_TYPE = unsigned int, typename HASH_TYPE = unsigned long long int>
struct TitanicIOBuffer {

	typedef ITER_TYPE IterType;
	typedef HASH_TYPE HashType;

	bool  mode;

	struct TitanicLink {
		IterType next;
		HashType hash;
		T        data;
	};

	IterType capacity;
	IterType overflow;

	AtomicIter<IterType> pull_exit_iter;


	__device__ ArrayIter<T,IterType> pull_span(IterType pull_size){
		Iter<IterType> pull_iter(0,0,0);
		if( mode ){
			pull_iter = iter.leap(pull_size);
		}
		return ArrayIter<T,IterType>(data,pull_iter);
	}

	__device__ ArrayIter<T,IterType> push(T value, unsigned int index, unsigned int priority){
		Iter<IterType> push_iter(0,0,0);
		if( !mode ){
			push_iter = iter.leap(push_size);
		}
		return ArrayIter<T,IterType>(data,push_iter);
	}

	__device__ void flip(){
		mode = !mode;	
		if(mode){
			iter = AtomicIter<IterType>(0,iter.value);
		} else {
			iter = AtomicIter<IterType>(0,0);
		}
	}

	__device__ bool empty() {
		return iter.done();
	}
	
};
#endif






void check_error(){

	cudaError_t status = cudaGetLastError();

	if(status != cudaSuccess){
		const char* err_str = cudaGetErrorString(status);
		printf("ERROR: \"%s\"\n",err_str);
	}

}



enum class GraphMode { Ascii, Braille, Block1x1, Block2x2, Block3x3, Slope };


struct GraphShape {

	int width;
	int height;
	float low;
	float high;


};


struct GraphConfig {

	GraphShape shape;
	GraphMode  mode;

};



void cli_graph(float* data, int size, int width, int height, float low, float high){




	#if GRAPH_MODE == 4
	const int   cols = 5;
	const int   rows = 5;
	const char* lookup[25] = {
		"⠀","⡀","⡄","⡆","⡇",
		"⢀","⣀","⣄","⣆","⣇",
		"⢠","⣠","⣤","⣦","⣧",
		"⢰","⣰","⣴","⣶","⣷",
		"⢸","⣸","⣼","⣾","⣿"
	};
	#elif GRAPH_MODE == 1
	const char* lookup[9] = {
		" ","▖","▌",
		"▗","▄","▙",
		"▐","▟","█"
	};
	#else
	const char* lookup[25] = {
		" ",".",".","i","|",
		".","_","_","L","L",
		".","_","o","b","b",
		"i","j","d",":","%",
		"|","J","d","4","#"
	};
	#endif

	
	float max = 0;
	for( int i=0; i<size; i++){
		if( data[i] > max ){
			max = data[i];
		}
	}

	printf("Max is %f\n",max);

	int x_iter;
	float l_val, r_val;
	float last=0;

	printf("%7.5f_\n",max);
	for(int i=0; i<height; i++){
		float base = (height-i-1)*max/height;
		printf("%7.5f_",base);
		x_iter = 0;
		for(int j=0; j<width; j++){
			l_val = 0;
			r_val = 0;
			int l_limit = (j*2*size)/(width*2);
			int r_limit = ((j*2+1)*size)/(width*2);
			float count = 0.0;
			for(; x_iter < l_limit; x_iter++){
				l_val += data[x_iter];
				//printf("%f,",data[x_iter]);
				count += 1.0;
			}
			l_val = ( count == 0.0 ) ? last : l_val / count;
			last = l_val;
			count = 0.0;
			for(; x_iter < r_limit; x_iter++){
				r_val += data[x_iter];
				count += 1.0;
			}
			r_val = ( count == 0.0 ) ? last : r_val / count;
			last = r_val;
			l_val = ( l_val - base )/max*height*4;
			r_val = ( r_val - base )/max*height*4;
			int l_idx = (l_val <= 0.0) ? 0 : ( (l_val >= 4.0) ? 4 : l_val );
			int r_idx = (r_val <= 0.0) ? 0 : ( (r_val >= 4.0) ? 4 : r_val );
			int str_idx = r_idx*5+l_idx;
			/*
			if( (str_idx < 0) || (str_idx >= 25) ){
				printf("BAD! [%d](%f:%d,%f:%d) -> (%d)",j,l_val,l_idx,r_val,r_idx,str_idx);
			}
			*/
			printf("%s",lookup[str_idx]);
		}
		printf("\n");
	}

	int   rule_size = 8*width/2;
	char* rule_vals = new char[rule_size]; 
	memset(rule_vals,'\0',rule_size);

	printf("        ");
	for(int j=0; j<width; j+=2){
		float l_limit = low + ((high-low)/width)*j;
		sprintf(&rule_vals[(8*j/2)],"%7.3f",l_limit);
		printf("\\ ");
	}
	printf("\n");
	for(int i=0; i<7; i++){
		printf("        ");
		for(int j=0; j<width; j+=2){
			printf(" %c",rule_vals[(8*j/2)+i]);
		}
		printf("\n");
	}

	free(rule_vals);

}




struct ArgSet
{

	int    argc;
	char** argv;

	int get_flag_idx(char* flag){
		for(int i=0; i<argc; i++){
			char* str = argv[i];
			if(    (str    != NULL )
			    && (str[0] == '-'  )
			    && (strcmp(str+1,flag) == 0)
                        ){
				return i;
			}
		}
		return -1;
	}


	char* get_flag_str(char* flag){
		int idx = get_flag_idx(flag);
		if( idx == -1 ) {
			return NULL;
		} else if (idx == (argc-1) ) {
			return (char*) "";
		}
		return argv[idx+1];
	}



	template<typename T>
	struct ArgVal {

		T     value;

		ArgVal(T val) : value(val)  {}
		
		ArgVal<T> operator| (T other) {
			return *this;
		}

		operator T() const {
			return value;
		}

	};


	struct ArgQuery {

		char* flag_str;
		char* value_str;


		template<typename T>
		bool scan_arg(char *str, T &dest) const {
			return false;
		}

		bool scan_arg(char *str, unsigned int &dest) const {	
			return ( 0 < sscanf(str,"%u",&dest) );
		}
		
		bool scan_arg(char *str, int &dest) const {	
			return ( 0 < sscanf(str,"%d",&dest) );
		}
		
		bool scan_arg(char *str, float &dest) const {	
			return ( 0 < sscanf(str,"%f",&dest) );
		}
		
		bool scan_arg(char *str, bool &dest) const{
			if        ( strcmp(str,"false") == 0 ){
				dest = false;
			} else if ( strcmp(str,"true" ) == 0 ){
				dest = true;
			} else {
				return false;
			}
			return true;
		}

		template<typename T>
		void scan_or_fail(T& dest) const{
			if(value_str == NULL) {
				printf("No value provided for flag '-%s'\n", flag_str);
				std::exit(1);
			}
			if( !scan_arg(value_str,dest) ){
				printf("Value string '%s' provided for flag '-%s' "
					"could not be parsed\n",
					value_str,flag_str
				);
				std::exit(1);
			}
		}

		void scan_or_fail(bool& dest) const{
			dest = (value_str != NULL);
		}

		template<typename T>
		ArgVal<T> operator| (T other) {
			if(value_str == NULL){
				return ArgVal<T>(other);
			} else {
				T value;
				scan_or_fail(value);
				return ArgVal<T>(value);
			}
		}

		ArgQuery operator| (ArgQuery other) {
			if(value_str == NULL){
				return other;
			} else {
				return *this;
			}
		}


		template<typename T>
		operator T() const {
			T value;
			scan_or_fail(value);
			return value;
		}


		ArgQuery(char* f, char* v) : flag_str(f), value_str(v) {}

	};


	ArgQuery operator[] (char* flag_str) {
		char* val_str = get_flag_str(flag_str);
		return ArgQuery(flag_str,val_str);
	}
	
	ArgQuery operator[] (const char* flag_str) {
		return (*this)[(char*)flag_str];
	}

	ArgSet(int c, char** v) : argc(c), argv(v) {}

};


struct Stopwatch {

	cudaEvent_t beg;
	cudaEvent_t end;
	float duration;

	Stopwatch() {
        	if (  ( cudaEventCreate( &beg ) != cudaSuccess )
		   || ( cudaEventCreate( &end  ) != cudaSuccess )
		) {
			printf("Failed to create Stopwatch\n");
			std::exit(1);
		}
	}


	bool start() {
		return ( cudaEventRecord( beg, NULL ) == cudaSuccess);
	}

	bool stop() {
		if ( cudaEventRecord( end, NULL ) != cudaSuccess ){
			return false;
		}
		cudaEventSynchronize( end );
        	cudaEventElapsedTime( &duration, beg, end );
		return true;
	}

	float ms_duration(){
		return duration;
	}

};





}




