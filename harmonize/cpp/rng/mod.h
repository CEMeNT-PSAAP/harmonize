#ifndef HARMONIZE_RNG
#define HARMONIZE_RNG

#include <limits>
#include <cstdint>

struct SimpleRNG {

    unsigned int state;

    SimpleRNG() = default;

    __host__ __device__ SimpleRNG(unsigned int state)
        : state(state)
    {}

    __host__ __device__ void advance_state () {
        state += 1;
    }

    template <typename T>
    __host__ __device__
    T rng ();


};

template <>
__host__ __device__ uint32_t SimpleRNG::rng <uint32_t> () {
    advance_state();
    return state;
}

template <> __host__ __device__ uint8_t  SimpleRNG::rng <uint8_t>  () { return rng<uint32_t>(); }
template <> __host__ __device__ uint16_t SimpleRNG::rng <uint16_t> () { return rng<uint32_t>(); }
template <> __host__ __device__ int8_t   SimpleRNG::rng <int8_t>   () { return rng<uint32_t>(); }
template <> __host__ __device__ int16_t  SimpleRNG::rng <int16_t>  () { return rng<uint32_t>(); }
template <> __host__ __device__ int32_t  SimpleRNG::rng <int32_t>  () { return rng<uint32_t>(); }


template <>
__host__ __device__ uint64_t SimpleRNG::rng <uint64_t> ()
{
    uint64_t result  = rng<uint32_t>();
    result = (result<<32) | rng<uint32_t>();
    return result;
}

template <>
__host__ __device__ int64_t SimpleRNG::rng <int64_t> ()
{
    int64_t result  = rng<int32_t>();
    result = (result<<32) | rng<int32_t>();
    return result;
}

template <>
__host__ __device__ float SimpleRNG::rng <float> () {
    return ((float)rng<uint32_t>()) / (float)0xFFFFFFFFu;
}
template <>
__host__ __device__ double SimpleRNG::rng <double> () {
    return ((double)rng<uint32_t>()) / (double)0xFFFFFFFFu;
}


#endif
