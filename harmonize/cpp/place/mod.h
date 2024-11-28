#ifndef HARMONIZE_PLACE
#define HARMONIZE_PLACE

namespace place
{

    typedef unsigned short RankIndex;
    typedef unsigned short DeviceIndex;
    typedef unsigned short BlockIndex;
    typedef unsigned short ThreadIndex;

    RankIndex   const NULL_RANK   = AdrInfo<RankIndex>  ::null;
    DeviceIndex const NULL_DEVICE = AdrInfo<DeviceIndex>::null;
    BlockIndex  const NULL_BLOCK  = AdrInfo<BlockIndex> ::null;
    ThreadIndex const NULL_THREAD = AdrInfo<ThreadIndex>::null;

    enum DeviceType : unsigned short {
        NULL = 0;
        CPU  = 1,
        GPU  = 2,
    };

    class Place {

        RankIndex   rank_index;
        DeviceType  device_type;
        DeviceIndex device_index;
        BlockIndex  block_index;
        ThreadIndex thread_index;

        public:

        __host__ __device__
        static RankIndex get_rank_index()
        {
            return 0;
        }

        __host__ __device__
        static DeviceType get_device_type()
        {
            #ifdef __CUDA_ARCH__
            return DeviceType::GPU;
            #else
            return DeviceType::CPU;
            #endif
        }

        __host__ __device__
        static RankIndex get_device_index()
        {
            #ifdef __CUDA_ARCH__
            int result;
            adapt::GPUrtGetDevice(&result);
            return result
            #else
            return 0;
            #endif
        }

        __host__ __device__
        static BlockIndex get_block_index()
        {
            #ifdef __CUDA_ARCH__
            return blockIdx.x;
            #else
            return NULL_BLOCK;
            #endif
        }

        __host__ __device__
        static ThreadIndex get_thread_index()
        {
            #ifdef __CUDA_ARCH__
            return threadIdx.x;
            #else
            return 0;
            #endif
        }

        __host__ __device__
        static Place here()
        {
            return {
                get_rank_index(),
                get_device_type(),
                get_device_index(),
                get_block_index(),
                get_thread_index()
            };
        }

        __host__ __device__
        static Place same_block()
        {
            return {
                get_rank_index(),
                get_device_type(),
                get_device_index(),
                get_block_index(),
                NULL_THREAD
            };
        }

        __host__ __device__
        static Place same_device()
        {
            return {
                get_rank_index(),
                get_device_type(),
                get_device_index(),
                NULL_THREAD,
                NULL_THREAD
            };
        }

        __host__ __device__
        static Place same_device_type()
        {
            return {
                get_rank_index(),
                get_device_type(),
                NULL_DEVICE,
                NULL_THREAD,
                NULL_THREAD
            };
        }

        __host__ __device__
        static Place same_rank()
        {
            return {
                get_rank_index(),
                DeviceType::NULL,
                NULL_DEVICE,
                NULL_THREAD,
                NULL_THREAD
            };
        }


    };

} // namespace place

#endif