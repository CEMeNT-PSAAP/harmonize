#pragma once



#if   defined(__NVCC__)

    // Nothing needed


#elif defined(__HIP__)

    #define __syncwarp(mask) printf("__syncwarp is not supported on AMD.\n")
    #define __activemask()   0x3F
    #define __any_sync(x,y) 0x1

#elif defined(__HCC__)



#elif defined(__CUDACC__)

    // This branch is entered in hipify

#else

    #error "Platform not recognized. Valid platforms include CUDA and HIP."

#endif


