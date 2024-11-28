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

    enum DeviceTypeID : unsigned short
    {
        NULL = 0;
        CPU  = 1,
        GPU  = 2,
    };


    template <DeviceTypeID DEVICE_TYPE_ID>
    class DeviceType
    {
        public:
        static DeviceTypeID TYPE_ID = DEVICE_TYPE_ID
    };


    class Place
    {

        RankIndex   rank_index;
        DeviceTypeID  device_type;
        DeviceIndex device_index;
        BlockIndex  block_index;
        ThreadIndex thread_index;

        public:

        __host__ __device__
        RankIndex get_rank_index()
        {
            return rank_index;
        }

        __host__ __device__
        DeviceTypeID get_device_type()
        {
            return device_type;
        }

        __host__ __device__
        DeviceIndex get_device_index()
        {
            return device_index;
        }

        __host__ __device__
        BlockIndex get_block_index()
        {
            return block_index;
        }
        __host__ __device__
        ThreadIndex get_thread_index()
        {
            return thread_index;
        }


        __host__ __device__
        static RankIndex current_rank_index()
        {
            return 0;
        }

        __host__ __device__
        static DeviceTypeID current_device_type()
        {
            #ifdef __CUDA_ARCH__
            return DeviceTypeID::GPU;
            #else
            return DeviceTypeID::CPU;
            #endif
        }

        __host__ __device__
        static RankIndex current_device_index()
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
        static BlockIndex current_block_index()
        {
            #ifdef __CUDA_ARCH__
            return blockIdx.x;
            #else
            return NULL_BLOCK;
            #endif
        }

        __host__ __device__
        static ThreadIndex current_thread_index()
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
                current_rank_index(),
                current_device_type(),
                current_device_index(),
                current_block_index(),
                current_thread_index()
            };
        }

        __host__ __device__
        static Place same_block()
        {
            return {
                current_rank_index(),
                current_device_type(),
                current_device_index(),
                current_block_index(),
                NULL_THREAD
            };
        }

        __host__ __device__
        static Place same_device()
        {
            return {
                current_rank_index(),
                current_device_type(),
                current_device_index(),
                NULL_THREAD,
                NULL_THREAD
            };
        }

        __host__ __device__
        static Place same_device_type()
        {
            return {
                current_rank_index(),
                current_device_type(),
                NULL_DEVICE,
                NULL_THREAD,
                NULL_THREAD
            };
        }

        __host__ __device__
        static Place same_rank()
        {
            return {
                current_rank_index(),
                DeviceTypeID::NULL,
                NULL_DEVICE,
                NULL_THREAD,
                NULL_THREAD
            };
        }

        __host__ __device__
        static Place any_place()
        {
            return {
                NULL_RANK,
                DeviceTypeID::NULL,
                NULL_DEVICE,
                NULL_THREAD,
                NULL_THREAD
            };
        }


    };

    template <typename ITEM_TYPE, size_t THREAD_COUNT>
    class PerThreadArray
    {
        ITEM_TYPE data[THREAD_COUNT];
    };

    template <typename ITEM_TYPE, size_t BLOCK_COUNT>
    class PerBlockArray
    {
        ITEM_TYPE data[BLOCK_COUNT];
    };

    template <typename ITEM_TYPE, size_t DEVICE_COUNT>
    class PerDeviceArray
    {
        ITEM_TYPE data[DEICE_COUNT];
    };

    template <typename ITEM_TYPE, DeviceTypeID DEVICE_TYPE_VALUE>
    class DeviceTypeToTypeMapping
    {
        public:
        typedef ITEM_TYPE ItemType;
        static DeviceTypeID const DEVICE_TYPE = DEVICE_TYPE_VALUE;
    }

    template <typename CPU_ITEM_TYPE, typename GPU_ITEM_TYPE>
    class DeviceTypeIDSelector {
        CPU_ITEM_TYPE &cpu_item;
        GPU_ITEM_TYPE &gpu_item;
        public:

        DeviceTypeIDSelector<CPU_ITEM_TYPE,GPU_ITEM_TYPE>(CPU_ITEM_TYPE &cpu_item, GPU_ITEM_TYPE &gpu_item)
        {

        }
    };



} // namespace place

#endif