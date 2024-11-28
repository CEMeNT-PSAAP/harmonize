

namespace alloc
{


template <typename ALLOCATOR_TYPE>
__host__ __device__
void allocation_juggle_test (size_t min_size, size_t max_size)
{

}


template <typename ALLOCATOR_TYPE>
__host__ __device__
void linked_list_construction_test(size_t min_size, size_t max_size)
{

}


DEFINE_LAUNCH_GLUE(allocation_juggle_test)
DEFINE_LAUNCH_GLUE(linked_list_construction_test)


template<typename ARENA_TYPE, typename POOL_TYPE>
TestLaunchResult test_alloc(TestLaunchConfig config)
{

}



TestModule test_module (mem::test_module,"alloc");


} // namespace alloc

