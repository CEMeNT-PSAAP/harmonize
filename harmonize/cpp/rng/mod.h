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
uint32_t SimpleRNG::rng <uint32_t> () {
    advance_state();
    return state;
}

template <> uint8_t  SimpleRNG::rng <uint8_t> () { return rng<uint32_t>(); }
template <> uint16_t SimpleRNG::rng <uint16_t> () { return rng<uint32_t>(); }
template <> int8_t   SimpleRNG::rng <int8_t> () { return rng<uint32_t>(); }
template <> int16_t  SimpleRNG::rng <int16_t> () { return rng<uint32_t>(); }
template <> int32_t  SimpleRNG::rng <int32_t> () { return rng<uint32_t>(); }


template <> uint64_t SimpleRNG::rng <uint64_t> ()
{
    uint64_t result  = rng<uint32_t>();
    result = (result<<32) | rng<uint32_t>();
    return result;
}
template <> int64_t SimpleRNG::rng <int64_t> ()
{
    int64_t result  = rng<int32_t>();
    result = (result<<32) | rng<int32_t>();
    return result;
}

template <> float    SimpleRNG::rng <float> () {
    return ((float)rng<uint32_t>()) / 0xFFFFFFFFu;
}
template <> double   SimpleRNG::rng <double> () {
    return ((double)rng<uint32_t>()) / 0xFFFFFFFFu;
}


#endif
