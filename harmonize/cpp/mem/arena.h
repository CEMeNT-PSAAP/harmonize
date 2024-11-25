#ifndef HARMONIZE_MEM_ARENA
#define HARMONIZE_MEM_ARENA

#include "core.h"



// A struct that directly contains an arena with a size
// pre-determined at compile time
template <typename ...>
struct DirectArena;

template <typename ITEM_TYPE, typename ADR_TYPE, typename ARENA_SIZE>
struct DirectArena <ITEM_TYPE, ADR_TYPE, ARENA_SIZE> : Managed
{
    typedef ITEM_TYPE  ItemType;
    typedef ADR_TYPE   AdrType;
    typedef ARENA_SIZE ArenaSize;

    static const size_t SIZE = ArenaSize::VALUE;
	static const size_t MAX_SIZE = AdrInfo<ADR_TYPE>::null;
	ItemType arena[SIZE];

    __host__ __device__ AdrType adr_of(ItemType &item)
    {
        return &item - arena;
    }

	__host__ __device__ ItemType& operator[](AdrType adr)
    {
		return arena[adr];
	}

    __host__ __device__ size_t size()
    {
        return SIZE;
    }
};


// A struct referencing an arena which may be any size that
// can be represented by the provided address type
template <typename ...>
struct IndirectArena;

template <typename ITEM_TYPE, typename ADR_TYPE>
struct IndirectArena <ITEM_TYPE, ADR_TYPE> : Managed
{
	typedef ITEM_TYPE ItemType;
	typedef ADR_TYPE  AdrType;

	static const size_t MAX_SIZE = ADR_TYPE::null;
	size_t    arena_size;
	ItemType *arena;

    __host__ __device__ AdrType adr_of(ItemType &item)
    {
        return &item - arena;
    }

	__host__ __device__ ItemType& operator[](AdrType adr)
    {
		return arena[adr];
	}

    __host__ __device__ size_t size()
    {
        return arena_size;
    }

};

#endif
