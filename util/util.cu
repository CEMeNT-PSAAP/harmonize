#pragma once

#include "math.h"
#include <vector>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <limits>
#include <memory>



namespace util {


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
#if 1 
__device__ unsigned int random_uint(unsigned int &state){

	state = (0x10DCDu * state + 1u);
	return state;

}
__device__ unsigned long long int random_uint(unsigned long long int &state){

	state = ( 2971215073 * state + 12345u);
	return state;

}
#else
__device__ unsigned int random_uint(unsigned int& rand_state){

	rand_state = (1103515245u * rand_state + 12345u);// % 0x80000000;
	return rand_state;

}
#endif


#include "func.cu"
#include "host.cu"
#include "mem.cu"
#include "iter.cu"
#include "cli.cu"









}




