

#ifndef HARMONIZE_ADAPT
#define HARMONIZE_ADAPT


#if   defined(__NVCC__)

    // Nothing needed

#elif defined(__HIP_PLATFORM_NVIDIA__)

    // Nothing needed

#elif defined(__HIP_PLATFORM_AMD__)

    #define __syncwarp(mask) printf("__syncwarp is not supported on AMD.\n")
    #define __activemask()   0x3F

#else

    #error "Platform not recognized. Valid platforms include CUDA and HIP."

#endif


#endif

