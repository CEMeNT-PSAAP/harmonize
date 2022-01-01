

#include "neut_common.cpp"

using namespace util;


typedef mem::MemPool<Neutron,unsigned int> PoolType;
typedef mem::MemCache<PoolType,128>        CacheType;


struct ThreadState {

	unsigned int rand_state;

};


struct GroupState {

	#ifdef CACHE
	CacheType cache;
	#endif

};


typedef ProgramStateDef<SimParams,GroupState,ThreadState> ProgState;

enum class Fn { Neutron };

DEF_PROMISE_TYPE(Fn::Neutron, unsigned int);


#ifdef EVENT
typedef  EventProgram     < PromiseUnion<Fn::Neutron>, ProgState > ProgType;
#else
typedef  HarmonizeProgram < PromiseUnion<Fn::Neutron>, ProgState > ProgType;
#endif






DEF_ASYNC_FN(ProgType, Fn::Neutron, arg) {


	if( arg == mem::Adr<unsigned int>::null ){
		printf("{   Bad argument!   }");
		return;
	}


	Neutron n;

	n = (*device.neutron_pool)[arg];

	int result = 0;
	for(int i=0; i < device.horizon; i++){
		result = step_neutron(device,n);
		if( result != 0 ){
			break;
		}
	}

	#ifdef FILO
	unsigned int last = n.next;
	#endif

	for(int i=0; i<result; i++){
		Neutron new_neutron(n);

		#ifdef FILO
		new_neutron.next = last;
		#endif

		#ifdef CACHE
		unsigned int index = group.cache.alloc_index(thread.rand_state);
		#else
		unsigned int index = device.neutron_pool->alloc_index(thread.rand_state);
		#endif

		if( index != mem::Adr<unsigned int>::null ){

			#ifdef ALLOC_CHECK
			unsigned int old = atomicCAS(&(*device.neutron_pool)[index].checkout,0u,1u);
			if( old != 0 ){
				printf("\n{Bad fiss alloc %d at %d}\n",old,index);
			}
			#endif

			#ifdef FILO
			last = index;
			#endif

			(*device.neutron_pool)[index] = new_neutron;

			#ifdef FILO
			if( i == (result-1) ){
				ASYNC_CALL(Fn::Neutron,index);
			}
			#else

			ASYNC_CALL(Fn::Neutron,index);

			#endif
		} else {
			printf("{Fiss alloc fail}");
		}


	}



	if( result == 0 ) {
		(*device.neutron_pool)[arg] = n;
		ASYNC_CALL(Fn::Neutron,arg);
	}
	else {

		#ifdef FILO
		if( (result < 0) && (n.next != mem::Adr<unsigned int>::null) ){
			ASYNC_CALL(Fn::Neutron, n.next);
		}
		#endif
		#ifdef ALLOC_CHECK
		unsigned int old = atomicCAS(&(*device.neutron_pool)[arg].checkout,1u,0u);
		if( old != 1 ){
			printf("{Bad dealloc %d at %d}",old,arg);
		}
		#endif

		#ifdef CACHE
		group.cache.free(arg,thread.rand_state);
		#else
		device.neutron_pool->free(arg,thread.rand_state);
		#endif

	}

}


DEF_INITIALIZE(ProgType) {


	#ifdef CACHE
	group.cache.initialize(*device.neutron_pool);
	#endif

	thread.rand_state = blockDim.x * blockIdx.x + threadIdx.x;

}


DEF_FINALIZE(ProgType) {

	#ifdef CACHE
	group.cache.finalize(thread.rand_state);
	#endif

}


DEF_MAKE_WORK(ProgType) {

	unsigned int id;

	#ifdef EVENT
	float fill_frac = QUEUE_FILL_FRACTION(Fn::Neutron);
	if( fill_frac > 0.9 ){
		return false;
	}

	iter::Iter<unsigned int> iter = device.source_id_iter->leap(2u);

	#else

	iter::Iter<unsigned int> iter = device.source_id_iter->leap(8u);

	#endif



	while(iter.step(id)){
		Neutron n(id,0.0,0.0,0.0,0.0);

		#ifdef FILO
		n.next = mem::Adr<unsigned int>::null;
		#endif

		#ifdef CACHE
		unsigned int index = group.cache.alloc_index(thread.rand_state);
		#else
		unsigned int index = device.neutron_pool->alloc_index(thread.rand_state);
		#endif
		while(index == mem::Adr<unsigned int>::null){
			printf("{failed to allocate}");
			#ifdef CACHE
			index = group.cache.alloc_index(thread.rand_state);
			#else
			index = device.neutron_pool->alloc_index(thread.rand_state);
			#endif
		}
		
		if( (index != mem::Adr<unsigned int>::null) ) {
			(*device.neutron_pool)[index] = n;

			#ifdef ALLOC_CHECK
			unsigned int old = atomicCAS(&(*device.neutron_pool)[index].checkout,0u,1u);
			if( old != 0 ){
				printf("\n{Bad alloc %d at %d}\n",old,index);
			}
			#endif

			#ifdef EVENT
			IMMEDIATE_CALL(Fn::Neutron,index);
			#else
			ASYNC_CALL   (Fn::Neutron,index);
			#endif
		}
		
	}

	return !device.source_id_iter->done();

}



int main(int argc, char *argv[]){

	using host::check_error;

	cli::ArgSet args(argc,argv);

	unsigned int wg_count = args["wg_count"];

	CommonContext com(args);

	cudaDeviceSynchronize();
		
	check_error();

	#ifdef EVENT
	ProgType::Instance instance = ProgType::Instance(0x40000,com.params);
	#else 
	ProgType::Instance instance = ProgType::Instance(0x100000,com.params);
	#endif

	cudaDeviceSynchronize();
	check_error();

	#ifdef EVENT

	do {
		exec<ProgType>(instance,wg_count,24);
		cudaDeviceSynchronize();
		check_error();
	} while ( ! instance.complete() );

	#else	

	init<ProgType>(instance,wg_count);
	cudaDeviceSynchronize();
	check_error();
	int num = 0;
	do {
		exec<ProgType>(instance,wg_count,0x10000);//0x800);
		cudaDeviceSynchronize();
		check_error();
		//ProgType::runtime_overview(instance);
		num++;
	} while(! instance.complete() );
	printf("\nIter count is %d\n",num);

	#endif

	return 0;

}

