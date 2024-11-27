#ifndef HARMONIZE_MEM_CORE
#define HARMONIZE_MEM_CORE

#include <limits>

#include "../preamble/mod.h"

// Provides a size_t as a type so that it may be
// folded into typename... parameter packs
template<size_t VALUE_ARG>
struct Size
{
    static const size_t VALUE = VALUE_ARG;
};


// Used as a parent type for types which should be
// dynamically allocated with managed memory
struct ManagedStorage
{
	void *operator new(size_t len)
	{
		void *ptr;
		util::host::auto_throw(adapt::GPUrtMallocManaged(&ptr, len));
		util::host::auto_throw(adapt::GPUrtDeviceSynchronize());
		return ptr;
	}

	void operator delete(void *ptr)
	{
		util::host::auto_throw(adapt::GPUrtDeviceSynchronize());
		util::host::auto_throw(adapt::GPUrtFree(ptr));
	}
};


struct DeviceStorage
{
	void *operator new(size_t len)
	{
		void *ptr;
		util::host::auto_throw(adapt::GPUrtMalloc(&ptr, len));
		util::host::auto_throw(adapt::GPUrtDeviceSynchronize());
		return ptr;
	}

	void operator delete(void *ptr)
	{
		util::host::auto_throw(adapt::GPUrtDeviceSynchronize());
		util::host::auto_throw(adapt::GPUrtFree(ptr));
	}
};


struct HostStorage
{
	void *operator new(size_t len)
	{
		void *ptr = malloc(len);
		if (ptr == nullptr) {
			throw std::runtime_error("Failed to allocate storage on host.");
		}
		return ptr;
	}

	void operator delete(void *ptr)
	{
		free(ptr);
	}
};


// Used to find integral types that are twice
// the width of a given integral type
template<typename T>
struct PairEquivalent;

template<>
struct PairEquivalent<unsigned short>
{
	typedef unsigned int Type;
};


template<>
struct PairEquivalent<unsigned int>
{
	typedef unsigned long long int Type;
};


// Used to make a pair of integers accessible as
// an integer with twice the width of its constituents
template<typename T>
union PairPack
{

	typedef PairPack<T> Self;

	typedef typename PairEquivalent<T>::Type PackType;

	static const PackType RIGHT_MASK = std::numeric_limits<T>::max();
	static const size_t   HALF_WIDTH = std::numeric_limits<T>::digits;
	static const PackType LEFT_MASK  = RIGHT_MASK << (PackType) HALF_WIDTH;

	struct PairType {
		T left;
		T right;
	};


	PairType pair;
	PackType pack;

};



// Provides information about an integer type so that it can be
// used to store addresses referencing elements of an arena
template <typename ADR_TYPE>
struct AdrInfo
{
	static const ADR_TYPE null = std::numeric_limits<ADR_TYPE>::max();
};

template <typename BASE_TYPE>
struct AdrInfo<BASE_TYPE*>
{
	static const BASE_TYPE *null = nullptr;
};


#endif
