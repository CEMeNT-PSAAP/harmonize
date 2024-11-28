#ifndef HARMONIZE_TEST_MEM_POOL
#define HARMONIZE_TEST_MEM_POOL

#include <iostream>

namespace pool {


const size_t ARENA_SIZE      = 10000000;
const size_t POOL_SIZE       = 1000;
const size_t LOCAL_POOL_SIZE = 100;
const size_t CHECK_COUNT     = 10000;

typedef mem::DirectArena<
    mem::Node<int,unsigned int,mem::Size<2>>,
    unsigned int,
    mem::Size<ARENA_SIZE>,
    mem::ManagedStorage
> ArenaType;

typedef mem::DequePool<
    ArenaType,
    POOL_SIZE,
    mem::ManagedStorage
> PoolType;



template<typename ARENA_TYPE, typename POOL_TYPE>
__host__ __device__
void coinflip_alloc_test(
    ARENA_TYPE *arena,
    POOL_TYPE *free_pool,
    unsigned int *fail_count
) {

    bool passed = true;
    SimpleRNG rng(1234);
    // Used to track the values each allocated node should have
    int memo[LOCAL_POOL_SIZE];
    // Used to track the addresses of each allocated node
    unsigned int local_pool[LOCAL_POOL_SIZE];

    // Start with an empty pool
    for (size_t i=0; i<LOCAL_POOL_SIZE; i++) {
        memo[i] = -1;
        local_pool[i] = mem::AdrInfo<unsigned int>::null;
    }

    // Perform a series of allocations and deallocations
    for (unsigned int i=0; i<CHECK_COUNT; i++) {

        // Pick an element in the local pool to use
        unsigned int index = rng.template rng<unsigned int>() % LOCAL_POOL_SIZE;
        // If the element is a null address, allocate a node and assign its address,
        // otherwise free it.
        if ( local_pool[index] == mem::AdrInfo<unsigned int>::null ) {
            unsigned int adr = free_pool->take_index(rng);

            if (adr == mem::AdrInfo<unsigned int>::null) {
                printf("Failed to allocate any index.\n");
                passed = false;
                return;
            }

            //printf("Took address %d\n",adr);
            // Set node value to random value
            int value = rng.template rng<int>() % 1000;
            int &item = (*arena)[adr].data;
            // Check that the node was not in use
            if ( (item != -1) || (memo[index] != -1) ) {
                printf("Claimed address %d that is currently being used.\n",adr);
                passed = false;
                return;
            } else {
                item = value;
                memo[index] = value;
            }
            local_pool[index] = adr;
        } else {
            unsigned int adr = local_pool[index];
            //printf("Giving index %d\n",adr);
            // Set node value to random value
            int &item = (*arena)[adr].data;
            // Check that node has value matching memo
            if (item != memo[index]) {
                printf("Freed node with mismatched value %d != %d.\n",item,memo[index]);
                passed = false;
            } else {
                item = -1;
                memo[index] = -1;
            }
            local_pool[index] = mem::AdrInfo<unsigned int>::null;
            free_pool->give_index(rng,adr);
        }
    }

    for (size_t i=0; i<LOCAL_POOL_SIZE; i++) {
        if ( local_pool[i] != mem::AdrInfo<unsigned int>::null) {
            free_pool->give_index(rng,local_pool[i]);
        }
    }

    if(!passed){
        intr::atomic::add_system(fail_count,1u);
    }

}




template<typename ARENA_TYPE, typename POOL_TYPE>
__host__ __device__
void exhaustion_test(
    ARENA_TYPE *arena,
    POOL_TYPE *free_pool,
    int *free_count,
    unsigned int *fail_count
) {

    SimpleRNG rng(1234);
    int remaining = intr::atomic::add_system(free_count,-1)-1;
    unsigned int head = mem::AdrInfo<unsigned int>::null;
    if (remaining > 0) {
        unsigned int adr = free_pool->take_index(rng);
    } else {
        intr::atomic::add_system(free_count,1);
    }

}




DEFINE_LAUNCH_GLUE(coinflip_alloc_test)
DEFINE_LAUNCH_GLUE(exhaustion_test)

template<typename ARENA_TYPE, typename POOL_TYPE>
TestLaunchResult test_deque_pool(TestLaunchConfig config)
{

    unsigned int *fail_count;
    int *free_count;
    util::host::auto_throw(adapt::GPUrtMallocManaged(&fail_count,sizeof(unsigned int)));
    util::host::auto_throw(adapt::GPUrtMallocManaged(&free_count,sizeof(int)));
    *fail_count = 0;
    *free_count = ARENA_SIZE;

    ARENA_TYPE *arena = new ARENA_TYPE;
    POOL_TYPE  *free_pool = new POOL_TYPE(*arena);

    for (unsigned int i=0; i<ARENA_SIZE; i++) {
        (*arena)[i].data = -1;
    }

    free_pool->reset();

    coinflip_alloc_test_launch_glue<ARENA_TYPE,POOL_TYPE>::launch_verbose(
        config,
        arena,free_pool,fail_count
    );

    exhaustion_test_launch_glue<ARENA_TYPE,POOL_TYPE>::launch_verbose(
        config,
        arena,free_pool,free_count,fail_count
    );

    delete arena;
    delete free_pool;

    return TestLaunchResult(true);
}



TestModule test_module (mem::test_module,"pool");

template <typename... PARAMS>
struct RegisterDequePool {
    TestLaunchSet deque_pool_test_set (
        test_module,
        "deque_pool",
        {{"default",test_deque_pool<PARAMS>}},
        {
            {1,0,0},
            {32,0,0},
            {0,1,32},
            {0,32,1},
            {0,32,32},
            {32,32,32},
        }
    );
}


} // namespace pool


#endif
