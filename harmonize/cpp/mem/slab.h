

typedef unsigned long long int DefaultAllocMaskElem;
typedef unsigned long long int DefaultSlabAdr;

const size_t DEFAULT_SLAB_SIZE = 1<<16;

// The largest contiguously allocatable unit of memory.
template <size_t SLAB_SIZE = DEFAULT_SLAB_SIZE, typename ELEM_TYPE = DefaultAllocMaskElem>
struct Slab
{
    typedef ELEM_TYPE AllocMaskElem;

    static const size_t ELEM_SIZE  = sizeof(ELEM_TYPE);
    static const size_t SIZE       = SLAB_SIZE;
    static const size_t ELEM_COUNT = (SLAB_SIZE+ELEM_SIZE-1)/ELEM_SIZE;

    // The actual storage provided by the slab
    ELEM_TYPE data[ELEM_COUNT];
};

// Is the default mechanism for managing allocations within each slab.
// This is kind of analogous to the slabinfo structs used by Linux, but
// it bundles the management logic as methods of this type. This allows
// developers to swap out their own intra-slab allocation logic and per-
// slab information as a single unit and allows the slab arena template
// to delegate the intra-slab allocation to the type.
template <typename SLAB_TYPE>
struct DefaultSlabProxy
{
    typedef SLAB_TYPE SlabType;
    typedef typename SLAB_TYPE::AllocMaskElem AllocMaskElem;

    typedef unsigned long long int AllocState;

    // In this implementation, the allocation state of a slab is tracked
    // through an allocation mask and an allocation state. The mask is
    // just an instance of the mask element type used by the underlying
    // slab type and is used to track a bitmask of which objects in the
    // slab are allocated. In cases where the number of objects stored in
    // the slab are not
    static AllocState const SIZE_OFFSET = sizeof(AllocState)*8/2;
    static AllocState const COUNT_MASK  = (((AllocState)1)<<SIZE_OFFSET)-(AllocState)1;
    static AllocState const SIZE_MASK   = ~COUNT_MASK;
    static size_t     const SLAB_SIZE   = SLAB_TYPE::SIZE;
    static size_t     const SLAB_ELEM_COUNT    = SLAB_TYPE::ELEM_COUNT;
    static size_t     const SLAB_ELEM_BIT_SIZE = sizeof(AllocMaskElem);

    // Bitfield used when the number of allocatable objects in the slab is <=64
    AllocMaskElem alloc_mask;
    // 32bit/32bit combination of slab's object size and allocation count.
    // This carries the implicit assumption that fewer than 2^(N/2) objects
    // can exist in a slab, with N being the bit width of the alloc state.
    AllocState alloc_state;

    // Extracts and returns the object size currently bound to the slab
    __host__ __device__
    size_t get_size()
    {
        return (alloc_state&SIZE_MASK)>>SIZE_OFFSET;
    }

    // Sets the color of the slab, and returns false if the slab was already bound
    // to a size, which would indicate a fundamental error in the allocator.
    __host__ __device__
    bool bind_size(unsigned int new_size)
    {
        AllocMaskElem long_size = new_size;
        AllocMaskElem prev = intr::atomic::CAS_system(&alloc_state,0llu,long_size<<SIZE_OFFSET);
        return (prev == 0);
    }

    // Clears the state of the slab, returning false if the slab did not have a non-zero size
    // or if it had a non-zero allocation count, either of which would indicate a fundamental
    // error in the allocator.
    __host__ __device__
    bool clear_alloc_state()
    {
        if (alloc_state == 0) {
            return false;
        }
        AllocMaskElem prev = intr::atomic::exch_system(&alloc_state,0llu);
        return ((prev&SIZE_MASK) != 0 ) && ((prev&COUNT_MASK) == 0);
    }

    // Returns the number of objects a slab may contain if it does not
    // contain an allocation mask.
    __host__ __device__
    size_t slab_object_count_without_alloc_mask(size_t object_size)
    {
        return SLAB_SIZE / object_size;
    }

    // Returns the number of objects a slab may contain if it does
    // contain an allocation mask.
    __host__ __device__
    size_t slab_object_count_with_alloc_mask(size_t object_size)
    {

        // Number of bits per object
        size_t object_bit_size = object_size*8;
        // Total number of bits that need to be stored in a slab per-object,
        // including the extra bit used in the allocation mask
        size_t bit_cost_per_object = (object_bit_size+1);
        // Total number of bits available per slab
        size_t total_bits = SLAB_ELEM_COUNT * SLAB_ELEM_BIT_SIZE;
        // The number of object bits we could ideally store if we didn't care about
        // whether or not we were storing fractions of an object or sharing elements
        // with the allocation mask. (TOBC = "total object bit count")
        size_t ideal_TOBC = (total_bits*object_bit_size) / bit_cost_per_object;
        // The maximum number of bits which can be used to store objects without
        // sharing elements with the alloc mask
        size_t max_exclusive_TOBC = ( ideal_TOBC / SLAB_ELEM_BIT_SIZE ) * SLAB_ELEM_BIT_SIZE;
        // The actual maximum number of whole objects we can store in the slab without
        // overlapping elements with the alloc mask
        size_t object_count = max_exclusive_TOBC / object_bit_size;
        return object_count;
    }

    // Returns the number of objects a slab can contain given an object size
    __host__ __device__
    size_t slab_object_count(size_t object_size)
    {
        size_t result = slab_object_count_without_alloc_mask(object_size);
        if (result > 64) {
            result = slab_object_count_with_alloc_mask(object_size);
        }
        return result;
    }

    // Clears the first `mask_elem_count` elements of the given slab
    __host__ __device__
    void clear_slab(SLAB_TYPE *slab, size_t mask_elem_count)
    {
        for (size_t i=0; i<mask_elem_count; i++) {
            slab->data[i] = 0;
        }
    }

    // Attempts to claim a slab for a given object size. Returns true if and
    // only if the claim was successful.
    __host__ __device__
    bool claim(SLAB_TYPE *slab, size_t object_size)
    {
        // Generate the state the alloc state should be in if initialized
        // for the given size with no outstanding allocations
        AllocState starting_state = ((AllocState)object_size)<<SIZE_OFFSET;

        // Attempt to swap in the initial state, returning failure if the
        // slab was somehow already claimed or otherwise in some bogus state.
        if (intr::atomic::CAS_system(&alloc_state,(AllocState)0,starting_state) != 0) {
            return false;
        }

        // Get the number of objects that may be held by the slab
        size_t object_count = slab_object_count(object_size);
        // If the slab cannot contain the given object size, return failure.
        if (object_count == 0) {
            return false;
        }

        // Clear the allocation mask, clearing the in-slab allocation mask if
        // it should exist.
        alloc_mask = 0;
        if (object_count > SLAB_ELEM_BIT_SIZE) {
            size_t mask_elem_count = (object_count+SLAB_ELEM_BIT_SIZE-1)/SLAB_ELEM_BIT_SIZE;
            clear_slab(slab,mask_elem_count);
        }
        return true;
    }

    // Attempts to atomically flip a bit in an allocation mask element, returning
    // the index of the bit if successful and otherwise returning false
    __host__ __device__
    bool attempt_mask_alloc(AllocMaskElem &elem, size_t &result)
    {
        // Just load the alloc mask element as a first guess
        AllocMaskElem mask_copy = alloc_mask;
        // Fail out if the mask becomes completely saturated (this should not
        // happen if the count in the alloc state is actually respected).
        while (intr::bitwise::population_count(mask_copy) < SLAB_ELEM_BIT_SIZE) {
            // Attempt to atomically set lowest unset bit in the mask
            size_t target = intr::bitwise::first_set(~mask_copy);
            AllocMaskElem target_mask = 1 << target;
            mask_copy = intr::atomic::or_system(&alloc_mask,target_mask);
            // If the bit wasn't already set, the allocation is successful
            if ((mask_copy&target_mask) == 0) {
                result = target;
                return true;
            }
        }
        return false;
    }


    // Gets the address of the `index`-th object in a slab, given a mask element count
    // and an object size
    __host__ __device__
    void *index_to_pointer(SLAB_TYPE *slab, size_t object_size, size_t mask_elem_count, size_t index)
    {
        // Convert slab pointer to something that can be offset byte-wise
        char *byte_ptr = static_cast<char*>(static_cast<void*>(slab));
        // Offset pointer past the last in-slab mask element
        byte_ptr += sizeof(AllocMaskElem) * mask_elem_count;
        // Offset pointer to the `index`-th object
        byte_ptr += object_size * index;
        // Convert back to a generic pointer
        return static_cast<void*>(byte_ptr);
    }


    // Attempts to claim a single object from the provided slab.
    __host__ __device__
    void *alloc(SLAB_TYPE *slab)
    {
        // Add one to the allocation count. As long as enough bits of wiggle room
        // exist for the count half of the alloc state, this should be safe.
        AllocMaskElem prev = intr::atomic::add_system(&alloc_state,(AllocMaskElem)1);

        // Extract count and size from previous allocation state
        AllocMaskElem prev_count = prev & COUNT_MASK;
        AllocMaskElem object_size  = (prev & SIZE_MASK) >> SIZE_OFFSET;

        // If the previous object count was already at or beyond the capacity of the
        // slab, roll back the incrementation.
        AllocMaskElem max_object_count = slab_object_count(object_size);
        if (prev_count >= max_object_count) {
            intr::atomic::add_system(&alloc_state,((AllocState)0)-((AllocState)1));
            return nullptr;
        }

        // Perform allocation in the proxy if the alloc mask can fit within the
        // provided mask, otherwise sweeping through the in-slab alloc mask
        size_t mask_elem_count = (max_object_count+SLAB_ELEM_BIT_SIZE-1)/SLAB_ELEM_BIT_SIZE;
        if (max_object_count <= SLAB_ELEM_BIT_SIZE) {
            size_t index;
            if ( attempt_mask_alloc(alloc_mask,index) ) {
                return index_to_pointer(slab,object_size,mask_elem_count,index);
            } else {
                return nullptr;
            }
        } else {
            // We have no easy fail out for the in-slab mask, since we cannot read
            // all elements of the mask at once and so cannot determine for certain
            // that the entire thing has been saturated. This should not be necessary
            // if the count in the alloc state is actually respected and memory
            // corruption does not occur. We'll just set a high loop limit and fail
            // out to prevent stalling.
            for (size_t i=0; i<2048; i++) {
                for (size_t j=0; j<SLAB_ELEM_COUNT; j++) {
                    size_t index;
                    if ( attempt_mask_alloc(slab->data[j],index) ) {
                        size_t full_index = SLAB_ELEM_BIT_SIZE*j + index;
                        return index_to_pointer(slab,object_size,mask_elem_count,full_index);
                    }
                }
            }
            return nullptr;
        }
    }

    // Attempts to free the object referenced by the provided pointer, returning false
    // upon failure.
    __host__ __device__
    bool free(SlabType *slab, void *obj_ptr)
    {

        // Find byte offset of object pointer relative to the slab
        char *obj_byte_ptr  = static_cast<char*>(obj_ptr);
        char *slab_byte_ptr = static_cast<char*>(static_cast<void*>(slab));
        size_t byte_offset = obj_byte_ptr - slab_byte_ptr;

        // Attempt to decrement the slab's allocation count
        AllocMaskElem prev = intr::atomic::add_system(&alloc_state,((AllocState)0)-((AllocState)1));
        AllocMaskElem prev_count = prev & COUNT_MASK;
        // If the previous count was zero (which it should never be if there is an
        // outstanding allocation) something bad has happened. Roll back the decrement
        // and fail out.
        if (prev_count == 0) {
            intr::atomic::add_system(&alloc_state,1llu);
            return false;
        }

        size_t object_size = (prev&SIZE_MASK)>>SIZE_OFFSET;

        // Find the upper bound on the intra-slab object index
        size_t max_object_count = slab_object_count(object_size);

        // Find the byte offset immediately after the last mask element
        size_t mask_count = (max_object_count+sizeof(AllocMaskElem)-1)/sizeof(AllocMaskElem);
        size_t mask_size  = mask_count * sizeof(AllocMaskElem);
        size_t first_obj_offset = byte_offset - mask_size;

        // Find the object index
        size_t obj_offset = byte_offset - first_obj_offset;
        // If pointer is not aligned, fail out
        if (obj_offset % sizeof(AllocMaskElem) != 0) {
            return false;
        }

        size_t obj_index = byte_offset / sizeof(AllocMaskElem);
        // If object index not in valid range, fail out
        if ((obj_index<0) || (obj_index >= max_object_count)) {
            return false;
        }


        // Attempt to clear corresponding bit in mask element
        size_t mask_index = obj_index / SLAB_ELEM_BIT_SIZE;
        size_t target_bit = obj_index % SLAB_ELEM_BIT_SIZE;
        AllocMaskElem target_mask = ((AllocMaskElem)1)<<((AllocMaskElem)target_bit);
        AllocMaskElem prev_mask = 0;
        if (mask_count<=SLAB_ELEM_BIT_SIZE) {
            prev_mask = intr::atomic::or_system(&alloc_mask,~target_mask);
        } else {
            prev_mask = intr::atomic::or_system(&(slab->data[mask_index]),~target_mask);
        }

        // Return success/failure depending upon if bit clearing is successful
        return ((prev_mask|target_mask) != 0);

    }



};


// Slab proxy template used to indicate no slab proxies should be used.
template <typename SLAB_TYPE>
struct EmptySlabProxy {};




// Bundles an arena of slabs with an arena of slab proxies held by nodes
template <
    typename ARENA_SIZE,
    template<typename> typename SLAB_PROXY_TYPE = DefaultSlabProxy,
    typename SLAB_TYPE       = Slab<DEFAULT_SLAB_SIZE>,
    typename SLAB_ADR_TYPE   = DefaultSlabAdr
>
class SlabArena
{

    typedef SLAB_ADR_TYPE              SlabAdrType;
    typedef SLAB_TYPE                  SlabType;
    typedef SLAB_PROXY_TYPE<SLAB_TYPE> SlabProxyType;
    typedef Node<SlabProxyType,SlabAdrType,Size<2>> ProxyNodeType;


    static size_t const SLAB_COUNT = (ARENA_SIZE::VALUE+SLAB_TYPE::SIZE-1)/SLAB_TYPE::SIZE;

    typedef DirectArena <
        SlabType,
        SlabAdrType,
        Size<SLAB_COUNT>
    > BackingArenaType;

    typedef DirectArena <
        ProxyNodeType,
        SlabAdrType,
        Size<SLAB_COUNT>
    > ProxyArenaType;

    BackingArenaType slabs;
    ProxyArenaType   proxies;

    public:

    // Get the index of the slab containing the given pointer
    __host__ __device__
    SlabAdrType slab_index_for(void *ptr) {
        char *byte_ptr = static_cast<char*>(ptr);
        char *base_byte_ptr = static_cast<char*>(static_cast<void*>(slabs.arena));

        size_t ptr_offset = byte_ptr - base_byte_ptr;
        size_t slab_index = ptr_offset / SlabType::SIZE;
        return slab_index;
    }

    // Get the slab containing the given pointer
    __host__ __device__
    SlabType &slab_for(void *ptr)
    {
        SlabAdrType slab_index = slab_index_for(ptr);
        return slabs.arena[slab_index];
    }

    // Get the proxy for the slab containing the given pointer
    __host__ __device__
    ProxyNodeType &proxy_for(void *ptr)
    {
        SlabAdrType slab_index = slab_index_for(ptr);
        return proxies[slab_index];
    }

};


// Bundles an arena of slabs with an arena of slab proxies held by nodes
template <
    typename ARENA_SIZE,
    typename SLAB_TYPE,
    typename SLAB_ADR_TYPE
>
class SlabArena <
    ARENA_SIZE,
    EmptySlabProxy,
    SLAB_TYPE,
    SLAB_ADR_TYPE
> {

    typedef SLAB_ADR_TYPE   SlabAdrType;
    typedef SLAB_TYPE       SlabType;

    static size_t const SLAB_COUNT = (ARENA_SIZE::VALUE+SLAB_TYPE::SIZE-1)/SLAB_TYPE::SIZE;

    typedef DirectArena <
        SlabType,
        SlabAdrType,
        Size<SLAB_COUNT>
    > BackingArenaType;

    BackingArenaType   slabs;

    public:

    // Get the index of the slab containing the given pointer
    __host__ __device__
    SlabAdrType slab_index_for(void *ptr) {
        char *byte_ptr = static_cast<char*>(ptr);
        char *base_byte_ptr = static_cast<char*>(static_cast<void*>(slabs.arena));

        size_t ptr_offset = byte_ptr - base_byte_ptr;
        size_t slab_index = ptr_offset / SlabType::SIZE;
        return slab_index;
    }


    // Get the slab containing the given pointer
    __host__ __device__
    SlabType &slab_for(void *ptr)
    {
        SlabAdrType slab_index = slab_index_for(ptr);
        return slabs.arena[slab_index];
    }

};


// Allocates slabs from a slab arena
template <typename ARENA_TYPE, size_t POOL_SIZE>
class SlabAllocator {

    typedef ARENA_TYPE                           ArenaType;
    typedef typename ArenaType::BackingArenaType BackingArenaType;
    typedef typename ArenaType::ProxyArenaType   ProxyArenaType;
    typedef typename ArenaType::SlabAdrType      SlabAdrType;
    typedef typename ArenaType::SlabType         SlabType;
    typedef typename ArenaType::ProxyNodeType    ProxyNodeType;
    typedef DequePool<ProxyArenaType,POOL_SIZE>  PoolType;


    ArenaType  &arena;
    PoolType    pool;
    SlabAdrType first_claim_iterator;


    public:

    __host__ __device__ SlabAllocator<ARENA_TYPE,POOL_SIZE> (ArenaType& arena)
        : arena(arena)
        , pool(arena)
        , first_claim_iterator(0)
    {}

    // Allocates a slab from the arena
    __host__ __device__ SlabAdrType alloc() {
        if (first_claim_iterator < arena.size()) {
            SlabAdrType index = intr::atomic::add_system(&first_claim_iterator,(SlabAdrType)1);
            if (index < arena.size()) {
                return index;
            }
        }
        return pool.take();
    }


    __host__ __device__ void free(SlabAdrType slab_adr) {
        pool.give(slab_adr);
    }

    __host__ __device__ SlabAdrType slab_index_for(void *ptr) {
        return arena.slab_index_for(ptr);
    }

    __host__ __device__ SlabType slab_for(void *ptr) {
        return arena.slab_for(ptr);
    }

    __host__ __device__ ProxyNodeType &proxy_for(void *ptr) {
        return arena.proxy_for(ptr);
    }

};


template <typename SLAB_ALLOCATOR_TYPE, size_t POOL_SIZE>
class SizedAllocator {

    typedef SLAB_ALLOCATOR_TYPE                       SlabAllocatorType;
    typedef typename SlabAllocatorType::SlabType      SlabType;
    typedef typename SlabAllocatorType::SlabAdrType   SlabAdrType;
    typedef typename SlabAllocatorType::SlabProxyType SlabProxyType;
    typedef typename SlabAllocatorType::PoolType      PoolType;
    typedef typename SlabAllocatorType::AllocMaskElem AllocMaskElem;

    SlabAllocatorType &slab_allocator;
    size_t             object_size;
    PoolType           pool;

    public:

    __host__ __device__ SizedAllocator<SLAB_ALLOCATOR_TYPE,POOL_SIZE> (
        SlabAllocatorType &slab_allocator,
        size_t object_size
    )
        : slab_allocator(slab_allocator)
        , object_size(object_size)
    {}

    __host__ __device__ void *alloc () {
        void *result = pool.take_index();
        while (result == nullptr) {
            SlabAdrType slab_adr = slab_allocator.alloc();
            if (slab_adr == AdrInfo<SlabAdrType>::null) {
                return nullptr;
            }

            SlabType      &slab       = slab_allocator.proxy_for(slab_adr);
            SlabProxyType &slab_proxy = slab_allocator.proxy_for(slab_adr);

            if (!slab_info.bind_color(object_size)) {
                return nullptr;
            }

            slab_info.swap_head_index(0);
            link_up_slab(slab);
            result = cache[size_index].alloc();
        }
    }

    __host__ __device__ bool free (void *ptr) {
        pool.give_index(ptr);
        return false;
    }


};


template <typename SLAB_ALLOCATOR_TYPE, size_t POOL_SIZE, size_t ALLOC_LIMIT>
class GeneralAllocator {

    typedef SLAB_ALLOCATOR_TYPE SlabAllocatorType;
    typedef typename SlabAllocatorType::SlabType SlabType;
    typedef typename SlabAllocatorType::SlabInfoType SlabInfoType;
    typedef typename SlabAllocatorType::AllocMaskElem AllocMaskElem;
    typedef SizedAllocator<SLAB_ALLOCATOR_TYPE,POOL_SIZE> SizedAllocatorType;


    SlabAllocatorType  &slab_allocator;
    SizedAllocatorType  cache[ALLOC_LIMIT];


    static const size_t MAX_SIZE = 1 << ALLOC_LIMIT;
    static const size_t MIN_SIZE = 1;


    __host__ __device__ void *alloc (size_t size) {
        size_t alloc_size = size;
        if (alloc_size > MAX_SIZE) {
            return nullptr;
        }
        if (alloc_size < MIN_SIZE) {
            alloc_size = MIN_SIZE;
        }
        size_t scaled_alloc_size = (alloc_size+MIN_SIZE-1)/MIN_SIZE;

        size_t size_index = 64-intr::bitwise::leading_zeros((unsigned long long int) scaled_alloc_size);

        return cache[size_index].alloc();
    }

    __host__ __device__ bool free (void *ptr) {

        char *byte_ptr = static_cast<char*>(ptr);
        char *base_byte_ptr = static_cast<char*>(static_cast<void*>(slab_allocator.arena.arena));

        size_t ptr_offset = byte_ptr - base_byte_ptr;
        size_t slab_index = ptr_offset / SlabType::SIZE;

        size_t size = slab_allocator.proxy_for(ptr).get_size();
        size_t size_index = 64-intr::bitwise::leading_zeros((unsigned long long int) size);

        return cache[size_index].free(ptr);

    }


};





