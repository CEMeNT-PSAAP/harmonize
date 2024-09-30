#ifndef HARMONIZE_ATOMIC
#define HARMONIZE_ATOMIC


namespace atomic {


template<typename T>
__host__ __device__ T add_system(T* adr,T val) {
    #ifdef __CUDA_ARCH__
        return atomicAdd_system(adr,val);
    #else
        return __sync_fetch_and_add(adr,val);
    #endif
}

template<typename T>
__host__ __device__ T sub_system(T* adr,T val) {
    #ifdef __CUDA_ARCH__
        return atomicSub_system(adr,val);
    #else
        return __sync_fetch_and_sub(adr,val);
    #endif
}

template<typename T>
__host__ __device__ T and_system(T* adr,T val) {
    #ifdef __CUDA_ARCH__
        return atomicAnd_system(adr,val);
    #else
        return __sync_fetch_and_and(adr,val);
    #endif
}

template<typename T>
__host__ __device__ T or_system(T* adr,T val) {
    #ifdef __CUDA_ARCH__
        return atomicOr_system(adr,val);
    #else
        return __sync_fetch_and_or(adr,val);
    #endif
}

template<typename T>
__host__ __device__ T xor_system(T* adr,T val) {
    #ifdef __CUDA_ARCH__
        return atomicXor_system(adr,val);
    #else
        return __sync_fetch_and_xor(adr,val);
    #endif
}

template<typename T>
__host__ __device__ T min_system(T* adr,T val) {
    #ifdef __CUDA_ARCH__
        return atomicMin_system(adr,val);
    #else
        return __sync_fetch_and_min(adr,val);
    #endif
}

template<typename T>
__host__ __device__ T exch_system(T* adr,T val) {
    #ifdef __CUDA_ARCH__
        return atomicExch_system(adr,val);
    #else
        T result;
        __atomic_exchange(adr,val,&result,__ATOMIC_ACQ_REL);
    #endif
}


};




#endif
