
#ifndef HARMONIZE_ADAPT
#define HARMONIZE_ADAPT

#define __HIP_PLATFORM_AMD__




#if defined(__NVCC__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC__)

    namespace adapt {

    size_t const WARP_SIZE = 32;

    #define CUDA_TYPE_ALIAS(thing) using GPUrt ## thing = cuda ## thing;
        CUDA_TYPE_ALIAS( Error_t );
        CUDA_TYPE_ALIAS( Event_t );
        CUDA_TYPE_ALIAS( Stream_t );
    #undef CUDA_TYPE_ALIAS


    #define CUDA_CONST_ALIAS(thing) auto const GPUrt ## thing = cuda ## thing;
        CUDA_CONST_ALIAS( Success );
        CUDA_CONST_ALIAS( MemcpyDeviceToDevice );
        CUDA_CONST_ALIAS( MemcpyDeviceToHost );
        CUDA_CONST_ALIAS( MemcpyHostToDevice );
        CUDA_CONST_ALIAS( MemcpyDefault );
    #undef CUDA_CONST_ALIAS


    #define CUDA_FN_ALIAS(thing)                               \
    template<typename... ARGS>                                 \
    auto GPUrt ## thing (ARGS... args)                         \
        -> decltype(cuda ## thing (std::declval<ARGS>()...))   \
    {                                                          \
        return cuda ## thing(args...);                         \
    }

        CUDA_FN_ALIAS( Malloc );
        CUDA_FN_ALIAS( DeviceSynchronize );
        CUDA_FN_ALIAS( GetErrorString );
        CUDA_FN_ALIAS( EventCreate );
        CUDA_FN_ALIAS( EventRecord );
        CUDA_FN_ALIAS( EventSynchronize );
        CUDA_FN_ALIAS( EventElapsedTime );
        CUDA_FN_ALIAS( GetLastError );
        CUDA_FN_ALIAS( Memcpy );
        CUDA_FN_ALIAS( Memset );
        CUDA_FN_ALIAS( SetDevice );
        CUDA_FN_ALIAS( Free );

    #undef CUDA_FN_ALIAS


    #define FN_ALIAS(renamed,original)                    \
    template<typename... ARGS>                            \
    auto renamed (ARGS... args)                           \
        -> decltype(original (std::declval<ARGS>()...))   \
    {                                                     \
        return original(args...);                         \
    }

        #if    __CUDA_ARCH__ < 600
            FN_ALIAS(atomicExch_block,atomicExch)
            FN_ALIAS(atomicCAS_block,atomicCAS)
        #endif

    #undef FN_ALIAS

    }


#elif defined(__HIP_PLATFORM_AMD__)

    #include <hip/hip_runtime.h>


    namespace adapt {

    #define __syncwarp(mask) ;
    #define __activemask()   0x3F
    #define __any_sync(x,y) 0x1

    size_t const WARP_SIZE = 64;

    #define HIP_TYPE_ALIAS(thing) using GPUrt ## thing = hip ## thing;
        HIP_TYPE_ALIAS( Error_t );
        HIP_TYPE_ALIAS( Event_t );
        HIP_TYPE_ALIAS( Stream_t );
    #undef HIP_TYPE_ALIAS


    #define HIP_CONST_ALIAS(thing) auto const GPUrt ## thing = hip ## thing;
        HIP_CONST_ALIAS( Success );
        HIP_CONST_ALIAS( MemcpyDeviceToDevice );
        HIP_CONST_ALIAS( MemcpyDeviceToHost );
        HIP_CONST_ALIAS( MemcpyHostToDevice );
        HIP_CONST_ALIAS( MemcpyDefault );
    #undef HIP_CONST_ALIAS


    #define HIP_FN_ALIAS(thing)                                \
    template<typename... ARGS>                                 \
    auto GPUrt ## thing (ARGS... args)                         \
        -> decltype(hip ## thing (std::declval<ARGS>()...))    \
    {                                                          \
        return hip ## thing(args...);                          \
    }

        HIP_FN_ALIAS( Malloc );
        HIP_FN_ALIAS( DeviceSynchronize );
        HIP_FN_ALIAS( GetErrorString );
        HIP_FN_ALIAS( EventCreate );
        HIP_FN_ALIAS( EventRecord );
        HIP_FN_ALIAS( EventSynchronize );
        HIP_FN_ALIAS( EventElapsedTime );
        HIP_FN_ALIAS( GetLastError );
        HIP_FN_ALIAS( Memcpy );
        HIP_FN_ALIAS( Memset );
        HIP_FN_ALIAS( SetDevice );
        HIP_FN_ALIAS( Free );

    #undef HIP_FN_ALIAS

    }

#else

    #error "Platform not recognized. Valid platforms include CUDA and HIP."

#endif


#endif

