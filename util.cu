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
struct DevObj
{
	T* adr;

	__host__ DevObj<T>(){
		cudaMalloc( (void**) &adr,  sizeof(T)  );
	}

	__host__ DevObj<T>(size_t size){
		cudaMalloc( (void**) &adr,  sizeof(T)*size  );
	}

	__host__ ~DevObj<T>(){
		cudaFree(adr);
	}

};


template<typename T>
struct _DevVec_Inner
{
	size_t size;
	T* adr;

	__host__ _DevVec_Inner<T>(size_t s){
		size = s;
		cudaMalloc( (void**) &adr,  sizeof(T)*size  );
	}

	__host__ ~_DevVec_Inner<T>(){
		cudaFree(adr);
	}

	__host__ void resize(size_t s){
		T* new_adr = cudaMalloc(&adr, sizeof(T)*s);
		size_t copy_count = ( s < size ) ? s : size;
		cudaMemcpy(new_adr,adr,sizeof(T)*copy_count);
		cudaFree(adr);
		size = s;
		adr = new_adr;
	}

	__host__ void operator<<(std::vector<T> &other) {
		cudaMemcpy(adr,other.data(),sizeof(T)*size,cudaMemcpyHostToDevice);
	}

	__host__ void operator>>(std::vector<T> &other) {
		if( other.size() != size ){
			other.resize(size);
		}
		cudaMemcpy(other.data(),adr,sizeof(T)*size,cudaMemcpyDeviceToHost);
	}

};


template<typename T>
struct DevVec
{

	std::shared_ptr<_DevVec_Inner<T>> _inner;

	__host__ DevVec<T>(size_t s) {
		_inner = std::make_shared<_DevVec_Inner<T>>(s);
	}


	__host__ void resize(size_t s) {
		_inner->resize(s);
	}

	__host__ void operator<<(std::vector<T> &other) {
		(*_inner)<<other;
	}

	__host__ void operator>>(std::vector<T> &other) {
		(*_inner)>>other;
	}

	__host__ operator T* () const {
		return _inner->adr;
	}

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
struct BasicIter
{


	typedef ITER_TYPE IterType;

	IterType value;
	IterType limit;
	IterType width;


	__device__ BasicIter<ITER_TYPE> (IterType v, IterType l, IterType w)
	: value(v), limit(l), width(w) {}


	 __device__ bool step(IterType& iter_val){

		if( value >= limit ){
			return false;
		}

		iter_val = value;
		value += width;
		return true;

	}

};



template<typename ITER_TYPE>
struct GroupWorkIter
{


	typedef ITER_TYPE IterType;

	IterType start;
	IterType limit;
	IterType chunk;
	IterType chunk_limit;

	 __device__ void reset(IterType start_val, IterType limit_val) {
		__syncwarp();
		if( current_leader() ){
			start = start_val;
			limit = limit_val;
			chunk = 0;
			IterType iter_width = limit_val - start_val;
			chunk_limit = iter_width / blockDim.x;
			if( (iter_width % blockDim.x) != 0 ){
				chunk_limit += 1;
			}
		}
		__syncwarp();
	}


	 __device__ bool step(IterType& iter_val) {
		
		if( chunk <= chunk_limit ){
			IterType val = start + chunk*blockDim.x + threadIdx.x;	
			if ( val < limit ){
				iter_val = val;
			}
			__syncwarp();
			if( current_leader() ){
				chunk += 1;
			}
			__syncwarp();
			return val < limit;
		}
		return false;
		
	}


	template<size_t MULTIPLIER>
	 __device__ BasicIter<IterType> multi_step() {
	
		BasicIter<IterType> result(0,0,blockDim.x);
		if( chunk <= chunk_limit ){
			IterType start_val = start + chunk*blockDim.x + threadIdx.x;	
			IterType limit_val = start_val + MULTIPLIER*blockDim.x;
			if ( start_val < limit ){
				result.value = start_val;
			} else {
				result.value = limit;
			}
			if ( limit_val < limit ){
				result.limit = limit_val;
			} else {
				result.limit = limit;
			}
			__syncwarp();
			if( current_leader() ){
				chunk += MULTIPLIER;
			}
			__syncwarp();
		}
		return result;
		
	}


	 __device__ bool done() {
		
		return ( chunk > chunk_limit );
		
	}


};



struct GlobalTurnstile
{

	unsigned long long int counter;

	__host__ static void reset(GlobalTurnstile* turnstile) {
		
	}


	 __device__ bool cross() {
	
		unsigned long long int checkout_index = atomicAdd(&counter,1);

		if(checkout_index == (gridDim.x-1)){
			atomicExch(&counter,0); 
			return true;
		} else {
			return false;
		}
		
	}

};



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




