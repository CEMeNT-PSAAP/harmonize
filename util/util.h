
#ifndef HARMONIZE_UTIL
#define HARMONIZE_UTIL

#include "math.h"
#include <vector>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <limits>
#include <memory>
#include <stdexcept>



namespace util {


/*
// Gives a count of how many threads in the current warp with a lower warp-local id are currently
// active. This is useful for finding an index for a set of operations performed once by an
// unpredictable number of threads within a warp.
*/
__device__ unsigned int warp_inc_scan();


/*
// This function returns the number of currently active threads in a warp
*/
__device__ unsigned int active_count();


/*
// This returns true only if the current thread is the active thread with the lowest warp-local id.
// This is valuable for electing a "leader" to perform single-threaded work for a warp.
*/
__device__ bool current_leader();
__device__ unsigned int pop_count(unsigned int value);
__device__ unsigned long long int pop_count(unsigned long long int value);


__device__ unsigned int leading_zeros(unsigned int value);
__device__ unsigned long long int leading_zeros(unsigned long long int value);


/*
// A simple pseudo-random number generator. This algorithm should never be used for cryptography,
// it is simply used to generate numbers random enough to reduce collisions for atomic
// instructions performed to manage the runtime state.
*/
#if 1
__host__ __device__ unsigned int random_uint(unsigned int &state);
__host__ __device__ unsigned long long int random_uint(unsigned long long int &state);

#else
__device__ unsigned int random_uint(unsigned int& rand_state);
#endif


struct Stopwatch {

	cudaEvent_t beg;
	cudaEvent_t end;
	float duration;

	Stopwatch();
	bool start();
	bool stop();
	float ms_duration();

};


}


#include "host.h"

#include "mem.h"

#include "iter.h"

#include "cli.h"


#endif
