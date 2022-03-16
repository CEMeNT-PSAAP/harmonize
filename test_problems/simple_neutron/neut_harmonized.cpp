

#include "neut_common.cpp"


#ifdef TIMER
	#define beg_clock(idx) if(util::current_leader()) { group.time_totals[idx] -= clock64(); }
	#define end_clock(idx) if(util::current_leader()) { group.time_totals[idx] += clock64(); }
#else
	#define beg_clock(idx) ;
	#define end_clock(idx) ;
#endif




using namespace util;


#ifdef BY_REF
typedef mem::MemPool<Neutron,unsigned int> PoolType;
//typedef mem::MemCache<PoolType,16>        CacheType;
typedef mem::SimpleMemCache<PoolType,16>      CacheType;
#endif


struct ThreadState {

	unsigned int rand_state;

};


struct GroupState {

	#ifdef BY_REF
	#ifdef CACHE
	CacheType cache;
	#endif
	#endif

	#ifdef LEVEL_CHECK
	int mem_level;
	int mem_max;
	#endif

	#ifdef TIMER
	unsigned long long int time_totals[TIMER];
	#endif

};


typedef ProgramStateDef<SimParams,GroupState,ThreadState> ProgState;

enum class Fn { Neutron };

#ifdef BY_REF
DEF_PROMISE_TYPE(Fn::Neutron, unsigned int);
#else
DEF_PROMISE_TYPE(Fn::Neutron, Neutron);
#endif


#ifdef EVENT
typedef  EventProgram     < PromiseUnion<Fn::Neutron>, ProgState > ProgType;
#else
#ifdef BY_REF
typedef  HarmonizeProgram < PromiseUnion<Fn::Neutron>, ProgState, unsigned int, 16, 127, 127 > ProgType;
#else
//typedef  HarmonizeProgram < PromiseUnion<Fn::Neutron>, ProgState, unsigned int,  8, 31, 31 > ProgType;
typedef  HarmonizeProgram < PromiseUnion<Fn::Neutron>, ProgState, unsigned int,  8, 8191, 8191 > ProgType;
#endif
#endif





DEF_ASYNC_FN(ProgType, Fn::Neutron, arg) {

	end_clock(8);
	
	#ifdef BY_REF
	if( arg == mem::Adr<unsigned int>::null ){
		printf("{   Bad argument!   }");
		return;
	}

	Neutron n;
	n = (*device.neutron_pool)[arg];
	#else
	Neutron n = arg;
	#endif

	beg_clock(1);

	int result = 0;
	for(int i=0; i < device.horizon; i++){
		result = step_neutron(device,n);
		if( result != 0 ){
			break;
		}
	}
	
	end_clock(1);

	#ifdef FILO
	unsigned int last = n.next;
	#endif


	beg_clock(2);

	for(int i=0; i<result; i++){
		Neutron new_neutron = n.child();
		
		#ifdef LEVEL_CHECK
		int mlev = atomicAdd(&group.mem_level,1);
		atomicMax(&group.mem_max,mlev+1);
		#endif

		#ifdef BY_REF

		#ifdef FILO
		new_neutron.next = last;
		#endif

		#ifdef CACHE
		unsigned int index = group.cache.alloc_index(thread.rand_state);
		#else
		unsigned int index = *device.neutron_pool.alloc_index(thread.rand_state);
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
		#else
			ASYNC_CALL(Fn::Neutron,new_neutron);
		#endif


	}

	end_clock(2);


	beg_clock(3);

	if( result == 0 ) {
		#ifdef BY_REF
		(*device.neutron_pool)[arg] = n;
		ASYNC_CALL(Fn::Neutron,arg);
		#else
		ASYNC_CALL(Fn::Neutron,n);
		#endif
	}

	end_clock(3);	

	beg_clock(4);	

	#ifdef BY_REF
	if( result != 0 ) {

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
		*device.neutron_pool.free(arg,thread.rand_state);
		#endif

	}
	#endif

	#ifdef LEVEL_CHECK
	if( result != 0 ){
		int mlev = atomicAdd(&group.mem_level,-1);
	}
	#endif

	end_clock(4);

	beg_clock(8);


}


DEF_INITIALIZE(ProgType) {

	thread.rand_state = blockDim.x * blockIdx.x + threadIdx.x;

	#ifdef BY_REF
	#ifdef CACHE
	group.cache.initialize(*device.neutron_pool);
	#endif
	#endif

	#ifdef TIMER
	if( util::current_leader() ) {
		for(unsigned int i=0; i<TIMER; i++){
			group.time_totals[i] = 0;
		}
	}
	beg_clock(0);
	#endif

	#ifdef LEVEL_CHECK
	if( util::current_leader() ){
		group.mem_level = 0;
		group.mem_max   = 0;
	}
	__syncwarp();
	#endif

	beg_clock(8);
}


DEF_FINALIZE(ProgType) {

	end_clock(8);

	#ifdef BY_REF
	#ifdef CACHE
	group.cache.finalize(thread.rand_state);
	#endif
	#endif

	#ifdef LEVEL_CHECK
	__syncwarp();
	if( util::current_leader() ){
		atomicAdd(device.level_total,group.mem_max);
	}
	#endif

	#ifdef TIMER
	end_clock(0);
	if( util::current_leader() ) {
		for(unsigned int i=0; i<TIMER; i++){
			atomicAdd(&(device.timer[i]),group.time_totals[i]);
		}
	}
	#endif

}


DEF_MAKE_WORK(ProgType) {

	end_clock(8);


	unsigned int id;

	beg_clock(5);
	
	#ifdef EVENT
	float fill_frac = QUEUE_FILL_FRACTION(Fn::Neutron);
	if( fill_frac > 0.5 ){
		return false;
	}

	iter::Iter<unsigned int> iter = device.source_id_iter->leap(1u);

	#else

	#ifdef BY_REF
	iter::Iter<unsigned int> iter = device.source_id_iter->leap(1u);
	#else
	iter::Iter<unsigned int> iter = device.source_id_iter->leap(1u);
	#endif

	#endif

	end_clock(5);	


	beg_clock(9);
	while(iter.step(id)){

		beg_clock(12);

		beg_clock(11);
		Neutron n(id,0.0,0.0,0.0,0.0,1.0);
		end_clock(11);

		#ifdef FILO
		n.next = mem::Adr<unsigned int>::null;
		#endif

		beg_clock(6);
	
		#ifdef BY_REF	
		#ifdef CACHE
		unsigned int index = group.cache.alloc_index(thread.rand_state);
		#else
		unsigned int index = *device.neutron_pool->alloc_index(thread.rand_state);
		#endif
		#endif
		
		end_clock(6);

		#ifdef LEVEL_CHECK
		int mlev = atomicAdd(&group.mem_level,1);
		atomicMax(&group.mem_max,mlev+1);
		#endif

		#ifdef BY_REF
		if(index == mem::Adr<unsigned int>::null){
			printf("{failed to allocate}");
		}
		
		beg_clock(7);
		if( (index != mem::Adr<unsigned int>::null) ) {
			beg_clock(13);
			(*device.neutron_pool)[index] = n;
			end_clock(13);

			#ifdef ALLOC_CHECK
			unsigned int old = atomicCAS(&(*device.neutron_pool)[index].checkout,0u,1u);
			if( old != 0 ){
				printf("\n{Bad alloc %d at %d}\n",old,index);
			}
			#endif

			beg_clock(10);
			beg_clock(8);
			IMMEDIATE_CALL(Fn::Neutron,index);
			//ASYNC_CALL(Fn::Neutron,index);
			end_clock(8);
			end_clock(10);
		}
		end_clock(7);
		#else
		IMMEDIATE_CALL(Fn::Neutron,n);
		//ASYNC_CALL(Fn::Neutron,n);
		#endif
		
		
		end_clock(12);
		
	}
	end_clock(9);

	beg_clock(8);

	return !device.source_id_iter->done();

}



int main(int argc, char *argv[]){

	using host::check_error;

	cli::ArgSet args(argc,argv);

	unsigned int wg_count = args["wg_count"];

	unsigned int dev_idx  = args["dev_idx"] | 0;
	cudaSetDevice(dev_idx); 

	CommonContext com(args);

	cudaDeviceSynchronize();
		
	check_error();

	//#ifndef BY_REF
	unsigned int arena_size = args["pool"] | 0x8000000;
	//#endif

	//printf("Constructing instance...\n");
	#ifdef EVENT
	#ifdef BY_REF
	ProgType::Instance instance = ProgType::Instance(arena_size,com.params);
	#else 
	ProgType::Instance instance = ProgType::Instance(arena_size,com.params);
	#endif
	#else 
	#ifdef BY_REF
	ProgType::Instance instance = ProgType::Instance(arena_size/32u,com.params);
	#else
	ProgType::Instance instance = ProgType::Instance(arena_size/32u,com.params);
	#endif
	#endif
	//printf("Constructed instance.\n");

	cudaDeviceSynchronize();
	check_error();

	#ifdef LEVEL_CHECK
	int high_level = 0;
	#endif

	//printf("Executing...\n");
	#ifdef EVENT

	do {
		exec<ProgType>(instance,wg_count,1);
		cudaDeviceSynchronize();
		check_error();
		#ifdef LEVEL_CHECK
		int level;
		com.level_total >> level;
		high_level = (level > high_level) ? level : high_level;
		com.level_total << 0;
		#endif
		//printf("\n---\n");
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
		#ifdef LEVEL_CHECK
		int level;
		com.level_total >> level;
		high_level = (level > high_level) ? level : high_level;
		com.level_total << 0;
		#endif
		//ProgType::runtime_overview(instance);
		num++;
	} while(! instance.complete() );
	//printf("\nIter count is %d\n",num);
	//printf("Completed\n");

	#ifdef HRM_TIME
	printf("Instance times:\n");
	instance.print_times();
	#endif

	#endif

	#ifdef LEVEL_CHECK
	printf("%d",high_level);
	#endif

	return 0;

}

