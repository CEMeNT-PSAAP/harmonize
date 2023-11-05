#pragma once




#if defined(__NVCC__) || HIPIFY

	#include "includes.h"
	#include "adapt.h"

#elif defined(__HIP__)

	#include "includes.h.hip"
	#include "adapt.h.hip"

#endif


void throw_err(const char *message, cudaError_t err) {
	std::string text = message;
	const char* err_str = cudaGetErrorString(err);
	text += " - Error: '";
	text += err_str;
	text += "'\n";
	throw std::runtime_error(text);
}


void throw_on_error(const char *message, cudaError_t err) {
	if (err != cudaSuccess) {
		throw_err(message,err);
	}
}

void checked_device_sync(){
	throw_on_error("Failed to sync with device.",cudaDeviceSynchronize());
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


__device__ unsigned int pop_count(unsigned int value) {
	return __popc(value);
}
__device__ unsigned long long int pop_count(unsigned long long int value) {
	return __popcll(value);
}


__device__ unsigned int leading_zeros(unsigned int value) {
	return __clz(value);
}
__device__ unsigned long long int leading_zeros(unsigned long long int value) {
	return __clzll(value);
}


/*
// A simple pseudo-random number generator. This algorithm should never be used for cryptography,
// it is simply used to generate numbers random enough to reduce collisions for atomic
// instructions performed to manage the runtime state.
*/
__host__ __device__ unsigned int random_uint(unsigned int &state){

	state = (0x10DCDu * state + 1u);
	return state;

}
__host__ __device__ unsigned long long int random_uint(unsigned long long int &state){

	state = ( 2971215073 * state + 12345u);
	return state;

}


struct Stopwatch {

	cudaEvent_t beg;
	cudaEvent_t end;
	float duration;

	Stopwatch() {

		cudaError_t beg_err = cudaEventCreate( &beg );
		cudaError_t end_err = cudaEventCreate( &end );

		throw_on_error("Failed to create Stopwatch start event.",beg_err);
		throw_on_error("Failed to create Stopwatch end event.",end_err);

	}


	void start() {
		cudaError_t beg_err = cudaEventRecord( beg, NULL );
		throw_on_error("Failed to submit Stopwatch start event.",beg_err);
	}

	void stop() {
		cudaError_t end_err = cudaEventRecord( end, NULL );
		throw_on_error("Failed to submit Stopwatch end event.",end_err);

		cudaError_t sync_err = cudaEventSynchronize( end );
		throw_on_error("Failed to synchronize on StopWatch end event.",sync_err);
		
        cudaError_t time_err = cudaEventElapsedTime( &duration, beg, end );
		throw_on_error("Failed to query StopWatch duration.",time_err);
	}

	float ms_duration(){
		return duration;
	}

};