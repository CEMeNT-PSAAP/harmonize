#include "../../../harmonize/cpp/harmonize.h"


const size_t ARENA_SIZE      = 1000000;
const size_t POOL_SIZE       = 100;
const size_t LOCAL_POOL_SIZE = 10;
const size_t CHECK_COUNT     = 1000000;

typedef NodeArena<DirectArena,Node<int,unsigned int,Size<1>>,Size<ARENA_SIZE>> ArenaType;
typedef DequePool<ArenaType,POOL_SIZE> PoolType;


template<typename ARENA_TYPE, typename DEQUE_TYPE>
__host__ __device__ void random_deque(
    ARENA_TYPE *arena,
    unsigned int *fail_count
) {

    DEQUE_TYPE deque = DEQUE_TYPE::make_empty();


}



template<typename ARENA_TYPE, typename POOL_TYPE>
__host__ __device__ void coinflip_alloc_test(
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

    for (size_t i=0; i<LOCAL_POOL_SIZE; i++) {
        memo[i] = -1;
        local_pool[i] = AdrInfo<unsigned int>::null;
    }

    // Perform a series of allocations and deallocations
    for (unsigned int i=0; i<CHECK_COUNT; i++) {

        // Pick an element in the local pool to use
        unsigned int index = rng.template rng<unsigned int>() % LOCAL_POOL_SIZE;
        // If the element is a null address, allocate a node and assign its address,
        // otherwise free it.
        if ( local_pool[index] == AdrInfo<unsigned int>::null ) {
            unsigned int adr = free_pool->take_index(rng);
            // Set node value to random value
            int value = rng.template rng<int>() % 1000;
            int &item = (*arena)[adr].data;
            // Check that the node was not in use
            if ( (item != -1) || (memo[index] != -1) ) {
                printf("Claimed value that is currently being used.\n");
                passed = false;
            } else {
                item = value;
                memo[index] = value;
            }
            local_pool[index] = adr;
        } else {
            unsigned int adr = local_pool[index];
            int &item = (*arena)[adr].data;
            // Check that node has value matching memo
            if (item != memo[index]) {
                printf("Freed node with mismatched value %d != %d.\n",item,memo[index]);
                passed = false;
            } else {
                item = -1;
                memo[index] = -1;
            }
            local_pool[index] = AdrInfo<unsigned int>::null;
            free_pool->give_index(rng,adr);
        }
    }

    for (size_t i=0; i<LOCAL_POOL_SIZE; i++) {
        if ( local_pool[i] != AdrInfo<unsigned int>::null) {
            free_pool->give_index(rng,local_pool[i]);
        }
    }

    if(!passed){
        atomic::add_system(fail_count,1u);
    }
}


template<typename ARENA_TYPE, typename POOL_TYPE>
__host__ __device__ void exhaustion_test(
    ARENA_TYPE *arena,
    POOL_TYPE *free_pool,
    int *free_count,
    unsigned int *fail_count
) {

    SimpleRNG rng(1234);
    int remaining = atomic::add_system(free_count,-1)-1;
    unsigned int head = AdrInfo<unsigned int>::null;
    if (remaining >= 0) {
        unsigned int adr = free_pool->take_index(rng);

    } else {
        atomic::add_system(free_count,1);
    }

}


template<typename ARENA_TYPE, typename POOL_TYPE>
__host__ void cpu_test_deque_pool(
    ARENA_TYPE *arena,
    POOL_TYPE *free_pool,
    int *free_count,
    unsigned int *fail_count
) {
    coinflip_alloc_test(arena,free_pool,fail_count);
    exhaustion_test(arena,free_pool,free_count,fail_count);
}

template<typename ARENA_TYPE, typename POOL_TYPE>
__global__ void gpu_test_deque_pool(
    ARENA_TYPE *arena,
    POOL_TYPE *free_pool,
    int *free_count,
    unsigned int *fail_count
) {
    coinflip_alloc_test(arena,free_pool,fail_count);
    exhaustion_test(arena,free_pool,free_count,fail_count);
}





template<typename ARENA_TYPE, typename POOL_TYPE>
bool test_decque_pool(bool on_gpu)
{

    unsigned int *fail_count;
    int *free_count;
    cudaMallocManaged(&fail_count,sizeof(unsigned int));
    cudaMallocManaged(&free_count,sizeof(int));
    *fail_count = 0;
    *free_count = ARENA_SIZE;

    ARENA_TYPE *arena = new ARENA_TYPE;
    POOL_TYPE  *free_pool = new POOL_TYPE(*arena);

    for (unsigned int i=0; i<ARENA_SIZE; i++) {
        (*arena)[i].data = -1;
    }

    if (on_gpu) {
        gpu_test_deque_pool<<<32,32>>>(arena,free_pool,free_count,fail_count);
    } else {
        cpu_test_deque_pool(arena,free_pool,free_count,fail_count);
    }

    delete arena;
    delete free_pool;

}



int main() {

    test_decque_pool<ArenaType,PoolType>(false);
    test_decque_pool<ArenaType,PoolType>(true);
    return 0;
}

