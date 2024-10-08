
#ifndef HARMONIZE_UTIL
#define HARMONIZE_UTIL




namespace util {


/*
// Gives a count of how many threads in the current warp with a lower warp-local id are currently
// active. This is useful for finding an index for a set of operations performed once by an
// unpredictable number of threads within a warp.
*/
static __device__ unsigned int warp_inc_scan(){

	unsigned int active = __activemask();
	unsigned int keep = (1 << threadIdx.x) - 1;
	unsigned int scan = __popc(active & keep);
	return scan;

}


/*
// This function returns the number of currently active threads in a warp
*/
static __device__ unsigned int active_count(){
	return __popc(__activemask());
}


/*
// This returns true only if the current thread is the active thread with the lowest warp-local id.
// This is valuable for electing a "leader" to perform single-threaded work for a warp.
// (__ffs (find first set) indexes bits from one, hence the -1 )
*/
static __device__ bool current_leader(){
	return ((__ffs(__activemask())-1) == threadIdx.x);
}


static __device__ unsigned int pop_count(unsigned int value) {
	return __popc(value);
}
static __device__ unsigned long long int pop_count(unsigned long long int value) {
	return __popcll(value);
}


static __device__ unsigned int leading_zeros(unsigned int value) {
	return __clz(value);
}
static __device__ unsigned long long int leading_zeros(unsigned long long int value) {
	return __clzll(value);
}


/*
// A simple pseudo-random number generator. This algorithm should never be used for cryptography,
// it is simply used to generate numbers random enough to reduce collisions for atomic
// instructions performed to manage the runtime state.
*/
#if 1
static __host__ __device__ unsigned int random_uint(unsigned int &state){

	state = (0x10DCDu * state + 1u);
	return state;

}
static __host__ __device__ unsigned long long int random_uint(unsigned long long int &state){

	state = ( 2971215073 * state + 12345u);
	return state;

}
#else
static __device__ unsigned int random_uint(unsigned int& rand_state){

	rand_state = (1103515245u * rand_state + 12345u);// % 0x80000000;
	return rand_state;

}
#endif


struct Stopwatch {

	adapt::GPUrtEvent_t beg;
	adapt::GPUrtEvent_t end;
	float duration;

	Stopwatch() {

		adapt::GPUrtError_t beg_stat = adapt::GPUrtEventCreate( &beg );
		adapt::GPUrtError_t end_stat = adapt::GPUrtEventCreate( &end );

		if(beg_stat != adapt::GPUrtSuccess){
			const char* err_str = adapt::GPUrtGetErrorString(beg_stat);
			printf("Failed to create Stopwatch start event. ERROR: \"%s\"\n",err_str);
		}

		if(end_stat != adapt::GPUrtSuccess){
			const char* err_str = adapt::GPUrtGetErrorString(end_stat);
			printf("Failed to create Stopwatch end event. ERROR: \"%s\"\n"  ,err_str);
		}

		if( (beg_stat != adapt::GPUrtSuccess) || (end_stat != adapt::GPUrtSuccess) ) {
			printf("Failed to create one or more Stopwatch events\n");
			std::exit(1);
		}
	}


	bool start() {
		return ( adapt::GPUrtEventRecord( beg, nullptr ) == adapt::GPUrtSuccess);
	}

	bool stop() {
		if ( adapt::GPUrtEventRecord( end, nullptr ) != adapt::GPUrtSuccess ){
			return false;
		}
		auto _x = adapt::GPUrtEventSynchronize( end );
		auto _y = adapt::GPUrtEventElapsedTime( &duration, beg, end );
		return true;
	}

	float ms_duration(){
		return duration;
	}

};


}


#include "host.h"

#include "mem.h"

#include "iter.h"

#include "cli.h"


#endif
