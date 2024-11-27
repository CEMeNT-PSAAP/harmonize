#ifndef HARMONIZE_MEM_POOL
#define HARMONIZE_MEM_POOL

#include "node.h"
#include "core.h"

// An array of deques that collectively hold a set of a certain
// type of node
template <typename ARENA_TYPE, size_t POOL_SIZE, typename STORAGE_TYPE>
struct DequePool : STORAGE_TYPE
{

    typedef ARENA_TYPE                   ArenaType;
    typedef typename ArenaType::AdrType  AdrType;
    typedef typename ArenaType::ItemType NodeType;
    typedef typename NodeType::DataType  DataType;

	static const size_t SIZE = POOL_SIZE;

    ArenaType &arena;
    NodeDeque<NodeType> deques[SIZE];

    DequePool<ARENA_TYPE,POOL_SIZE,STORAGE_TYPE>() = default;

    __host__ __device__
    DequePool<ARENA_TYPE,POOL_SIZE,STORAGE_TYPE>(ARENA_TYPE& arena)
        : arena(arena)
    {}


    __host__ __device__
    void reset ()
    {

        #if __HARMONIZE_DEVICE_COMPILE__
            size_t arena_size = arena.size();
            size_t thread_count = gridDim.x * blockDim.x;
            size_t thread_id    = blockDim.x * blockIdx.x + threadIdx.x;
            size_t chunk_size   = (arena_size+SIZE-1) / SIZE;
            for (size_t i=thread_id; i<SIZE; i+=thread_count) {
                deques[i] = NodeDeque<NodeType>::make_empty();
                size_t start = std::min(chunk_size*i,    arena_size);
                size_t end   = std::min(chunk_size*(i+1),arena_size);
                NodeDequeProxy<ARENA_TYPE> proxy(arena,deques[i]);
                proxy.take();
                for (size_t j=start; j<end; j++) {
                    proxy.push(j);
                }
                proxy.give();
            }
        #else
            size_t arena_size = arena.size();
            size_t chunk_size = (arena_size+SIZE-1) / SIZE;
            for (size_t i=0; i<SIZE; i++) {
                deques[i] = NodeDeque<NodeType>::make_empty();
                size_t start = std::min(chunk_size*i,    arena_size);
                size_t end   = std::min(chunk_size*(i+1),arena_size);
                NodeDequeProxy<ARENA_TYPE> proxy(arena,deques[i]);
                proxy.take();
                for (size_t j=start; j<end; j++) {
                    proxy.push(j);
                }
                proxy.give();
            }
        #endif
    }

    template<typename STATE>
    __host__ __device__
    AdrType take_index(STATE &state)
    {
        size_t start = state.template rng<size_t>() % SIZE;
        size_t index = start;
        AdrType result = AdrInfo<AdrType>::null;
        do {
            // Scan for a non-empty deque
            NodeDequeProxy<ArenaType> deque_proxy(arena,deques[index]);
            deque_proxy.take();
            result = deque_proxy.pop();
            deque_proxy.give();
            index = (index+1) % SIZE;
        } while ( (index != start) && (result == AdrInfo<AdrType>::null));
        return result;
    }

    template<typename STATE>
    __host__ __device__
    NodeType *take(STATE &state)
    {
        return arena + take_index<STATE>(state);
    }

    template<typename STATE>
    __host__ __device__
    void give_index(STATE &state, AdrType adr)
    {
        size_t index = state.template rng<size_t>() % SIZE;
        NodeDequeProxy<ArenaType> deque_proxy(arena,deques[index]);
        deque_proxy.take();
        deque_proxy.push(adr);
        deque_proxy.give();
    }

    template<typename STATE>
    __host__ __device__
    void give(STATE &state, NodeType *ptr)
    {
        give_index(state,arena.offset_of(*ptr));
    }

};



// A DequePool that tracks the number of elements it contains. This container
// is less performant when there are many concurrent operations, since
// transactions are bottlenecked on checking the counter.
template <typename ARENA_TYPE, size_t POOL_SIZE, typename STORAGE_TYPE>
struct CountedDequePool : DequePool<ARENA_TYPE, POOL_SIZE, STORAGE_TYPE>
{

    typedef DequePool<ARENA_TYPE,POOL_SIZE,STORAGE_TYPE> Parent;
    typedef typename Parent::AdrType        AdrType;
    typedef typename Parent::NodeType       NodeType;

    long long int count;

    CountedDequePool<ARENA_TYPE,POOL_SIZE>() = default;

    __host__ __device__
    CountedDequePool<ARENA_TYPE,POOL_SIZE>(ARENA_TYPE& arena)
        : Parent(arena)
        , count(0)
    {}

    template<typename STATE>
    __host__ __device__
    AdrType take_index(STATE &state)
    {
       long long int remaining = intr::atomic::add_system(&count,-1ll)-1;
       if (remaining < 0) {
            intr::atomic::add_system(&count,1ll);
            return AdrInfo<AdrType>::null;
       }
       Parent::take_index(state);
    }

    template<typename STATE>
    __host__ __device__
    NodeType *take(STATE &state)
    {
        return Parent::arena + take_index<STATE>(state);
    }

    template<typename STATE>
    __host__ __device__
    void give_index(STATE &state, AdrType adr)
    {
        Parent::give_index(state);
        intr::atomic::add_system(&count,1ll);
    }

    template<typename STATE>
    __host__ __device__
    void give(STATE &state, NodeType *ptr)
    {
        give_index(state,Parent::arena.offset_of(*ptr));
    }

};



#endif
