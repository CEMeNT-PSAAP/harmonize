#ifndef HARMONIZE_MEM_NODE
#define HARMONIZE_MEM_NODE

#include "arena.h"

// A value which is stored with some number of addresses
// referencing adjacent values in the same arena
template <typename DATA_TYPE, typename ADR_TYPE, typename LINK_ARRAY_SIZE>
struct Node
{
	typedef ADR_TYPE   AdrType;
    typedef DATA_TYPE  DataType;
    typedef LINK_ARRAY_SIZE LinkArraySize;

	static const size_t LINK_COUNT = LinkArraySize::VALUE;

    DataType data;
    AdrType  link[LINK_COUNT];
};

template <typename ADR_TYPE, typename LINK_ARRAY_SIZE>
struct Node <void,ADR_TYPE,LINK_ARRAY_SIZE>
{
	typedef ADR_TYPE   AdrType;
    typedef LINK_ARRAY_SIZE LinkArraySize;

	static const size_t LINK_COUNT = LinkArraySize::VALUE;

    AdrType  link[LINK_COUNT];
};


// References a Node and its Arena, allowing for more
// ergonomic traversal
template <typename ARENA_TYPE>
struct NodeProxy
{

    typedef ARENA_TYPE ArenaType;
    typedef NodeProxy<ARENA_TYPE>        Self;
    typedef typename ArenaType::ItemType  NodeType;
    typedef typename NodeType::DataType   DataType;
    typedef typename NodeType::AdrType    AdrType;

    ArenaType &arena;
    NodeType   &node;

    __host__ __device__ NodeProxy<ARENA_TYPE> (
        ArenaType &arena,
        NodeType   &node
    )
        : arena(arena)
        , node(node)
    {}

    __host__ __device__ AdrType &link(size_t index)
    {
        return node.link[index];
    }

    __host__ __device__ Self adj(size_t index)
    {
        return {arena,arena[link(index)]};
    }

};


// References a Node and its Arena, allowing for more
// ergonomic traversal
template<typename DATA_TYPE, typename LINK_ARRAY_SIZE>
struct NodeProxy<Node<DATA_TYPE,void*,LINK_ARRAY_SIZE>*>
{

    typedef void*                                            ArenaType;
    typedef Node<DATA_TYPE,void*,LINK_ARRAY_SIZE>            NodeType;
    typedef NodeProxy<Node<DATA_TYPE,void*,LINK_ARRAY_SIZE>> Self;
    typedef typename NodeType::DataType                      DataType;
    typedef void*                                            AdrType;

    NodeType   &node;

    __host__ __device__ NodeProxy<void*> (
        NodeType   &node
    )
        : node(node)
    {}

    __host__ __device__ AdrType &link(size_t index)
    {
        return node.link[index];
    }

    __host__ __device__ Self adj(size_t index)
    {
        return {*(NodeType)link(index)};
    }

};

#ifdef HARMONIZE_PAIR_DEQUES

// Holds head and tail addresses to a linked list of Nodes,
// with addresses packed into a pair.
template <typename NODE_TYPE>
struct NodeDeque
{

    typedef NODE_TYPE                  NodeType;
    typedef typename NodeType::AdrType AdrType;

    PairPack<AdrType> pair;

    __host__ __device__ static NodeDeque make_empty()
    {
        return NodeDeque{PairPack<AdrType>{
            AdrInfo<AdrType>::null,
            AdrInfo<AdrType>::null
        }};
    }

    __host__ __device__ void swap(NodeDeque& other)
    {
        other.pair.pack = atomic::exch_system(&pair.pack,other.pair.pack);
    }

    __host__ __device__ AdrType &head()
    {
        return pair.pair.left;
    }

    __host__ __device__ AdrType &tail()
    {
        return pair.pair.right;
    }

    __host__ __device__ bool is_empty()
    {
        AdrType null  = AdrInfo<AdrType>::null;
        return ((head()==null) && (tail()==null));
    }

};


// Maintains references to a shared NodeDeque, the Arena that
// the NodeDeque references, and a local NodeDeque that can
// be safely manipulated.
template <typename ARENA_TYPE>
class NodeDequeProxy
{

    typedef ARENA_TYPE                    ArenaType;
    typedef NodeProxy<ARENA_TYPE>         Self;
    typedef typename ArenaType::ItemType  NodeType;
    typedef typename NodeType::DataType   DataType;
    typedef typename NodeType::AdrType    AdrType;
    typedef NodeDeque<NodeType>           DequeType;

    ArenaType &arena;
    DequeType &shared;
    DequeType local;

    public:


    __host__ __device__ NodeDequeProxy<ARENA_TYPE> (
        ArenaType &arena,
        DequeType &shared
    )
        : arena(arena)
        , shared(shared)
        , local(DequeType::make_empty())
    {}

    __host__ __device__ void merge(DequeType& other)
    {
        if (local.is_empty()) {
            local = other;
        } else if (!other.is_empty()) {
            NodeType &node = arena[local.tail()];
            node.link[0] = other.head();
            local.tail() = other.tail();
        }
        other = DequeType::make_empty();
    }


    __host__ __device__ void take()
    {
        DequeType temp = DequeType::make_empty();
        shared.swap(temp);
        merge(temp);
    }

    __host__ __device__ void give()
    {
        while (!local.is_empty()) {
            DequeType temp = DequeType::make_empty();
            shared.swap(temp);
            merge(temp);
            shared.swap(local);
        }
    }

    __host__ __device__ AdrType pop()
    {
        AdrType result = local.head();
        if (local.head() != AdrInfo<AdrType>::null) {
            local.head() = arena[local.head()].link[0];
        }
        return result;
    }

    __host__ __device__ void push(AdrType adr)
    {
        arena[adr].link[0] = local.head();
        local.head() = adr;
    }

};



#else

// Holds head and tail addresses to a linked list of Nodes,
// with addresses packed into a pair.
template <typename NODE_TYPE>
struct NodeDeque
{

    typedef NODE_TYPE                  NodeType;
    typedef typename NodeType::AdrType AdrType;

    AdrType head_index;

    __host__ __device__ static NodeDeque make_empty()
    {
        return NodeDeque{AdrInfo<AdrType>::null};
    }

    __host__ __device__ void swap(NodeDeque& other)
    {
        other.head_index = intr::atomic::exch_system(&head_index,other.head_index);
    }

    __host__ __device__ bool is_empty()
    {
        AdrType null  = AdrInfo<AdrType>::null;
        return head_index==null;
    }

};


// Maintains references to a shared NodeDeque, the Arena that
// the NodeDeque references, and a local NodeDeque that can
// be safely manipulated.
template <typename ARENA_TYPE>
class NodeDequeProxy
{

    typedef ARENA_TYPE                    ArenaType;
    typedef NodeProxy<ARENA_TYPE>         Self;
    typedef typename ArenaType::ItemType  NodeType;
    typedef typename NodeType::DataType   DataType;
    typedef typename NodeType::AdrType    AdrType;
    typedef NodeDeque<NodeType>           DequeType;

    ArenaType &arena;
    DequeType &shared;
    DequeType local;

    public:


    __host__ __device__ NodeDequeProxy<ARENA_TYPE> (
        ArenaType &arena,
        DequeType &shared
    )
        : arena(arena)
        , shared(shared)
        , local(DequeType::make_empty())
    {}

    __host__ __device__ void merge(DequeType& other)
    {
        if (local.is_empty()) {
            local = other;
        } else if (!other.is_empty()) {
            NodeType &head       = arena[local.head_index];
            NodeType &tail       = arena[head.link[1]];
            NodeType &other_head = arena[other.head_index];
            tail.link[0] = other.head_index;
            head.link[1] = other_head.link[1];
        }
        other = DequeType::make_empty();
    }


    __host__ __device__ void take()
    {
        DequeType temp = DequeType::make_empty();
        shared.swap(temp);
        merge(temp);
    }

    __host__ __device__ void give()
    {
        while (!local.is_empty()) {
            DequeType temp = DequeType::make_empty();
            shared.swap(temp);
            merge(temp);
            shared.swap(local);
        }
    }

    __host__ __device__ AdrType pop()
    {
        AdrType result = local.head_index;
        if (local.head_index != AdrInfo<AdrType>::null) {
            NodeType &old_head = arena[local.head_index];
            local.head_index = old_head.link[0];
            if (local.head_index != AdrInfo<AdrType>::null) {
                NodeType &new_head = arena[local.head_index];
                new_head.link[1] = old_head.link[1];
            }
        }
        return result;
    }

    __host__ __device__ void push(AdrType adr)
    {
        if (adr == AdrInfo<AdrType>::null) {
            return;
        }

        AdrType old_head_index = local.head_index;
        local.head_index = adr;

        NodeType &new_head = arena[local.head_index];

        if (old_head_index == AdrInfo<AdrType>::null) {
            new_head.link[1] = adr;
        } else {
            NodeType &old_head = arena[old_head_index];
            new_head.link[1] = old_head.link[1];
        }
        new_head.link[0] = old_head_index;
    }

};


#endif


#endif
