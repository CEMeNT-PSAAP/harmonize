#pragma once



#if   defined(__NVCC__)

    size_t const WARP_SIZE = 32;


#elif defined(__HIP__) || HIPIFY

    #define __syncwarp(mask) ;
    #define __activemask()   0x3F
    #define __any_sync(x,y) 0x1
    size_t const WARP_SIZE = 64;


#elif defined(__HCC__)



#elif defined(__CUDACC__)

    // This branch is entered in hipify

#else

    #error "Platform not recognized. Valid platforms include CUDA and HIP."

#endif


