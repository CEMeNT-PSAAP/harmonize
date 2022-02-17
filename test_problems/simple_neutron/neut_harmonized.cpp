

#include "neut_common.cpp"



using namespace util;


typedef mem::MemPool<Neutron,unsigned int> PoolType;
//typedef mem::MemCache<PoolType,16>        CacheType;
typedef mem::SimpleMemCache<PoolType,16>      CacheType;


struct ThreadState {

	unsigned int rand_state;

};


struct GroupState {

	#ifdef CACHE
	CacheType cache;
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

DEF_PROMISE_TYPE(Fn::Neutron, unsigned int);


#ifdef EVENT
typedef  EventProgram     < PromiseUnion<Fn::Neutron>, ProgState > ProgType;
#else
typedef  HarmonizeProgram < PromiseUnion<Fn::Neutron>, ProgState, unsigned int, 16, 64, 64 > ProgType;
#endif






DEF_ASYNC_FN(ProgType, Fn::Neutron, arg) {

	#ifdef TIMER
	if( util::current_leader() ) {
		group.time_totals[8] += clock64();
	}
	#endif

	if( arg == mem::Adr<unsigned int>::null ){
		printf("{   Bad argument!   }");
		return;
	}


	Neutron n;

	n = (*device.neutron_pool)[arg];

	#ifdef TIMER
	if( util::current_leader() ) {
		group.time_totals[1] -= clock64();
	}
	#endif

	int result = 0;
	for(int i=0; i < device.horizon; i++){
		result = step_neutron(device,n);
		if( result != 0 ){
			break;
		}
	}
	
	#ifdef TIMER
	if( util::current_leader() ) {
		group.time_totals[1] += clock64();
	}
	#endif

	#ifdef FILO
	unsigned int last = n.next;
	#endif


	#ifdef TIMER
	if( util::current_leader() ) {
		group.time_totals[2] -= clock64();
	}
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
		
		#ifdef LEVEL_CHECK
		int mlev = atomicAdd(&group.mem_level,1);
		atomicMax(&group.mem_max,mlev+1);
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

	#ifdef TIMER
	if( util::current_leader() ) {
		group.time_totals[2] += clock64();
	}
	#endif

	#ifdef TIMER
	if( util::current_leader() ) {
		group.time_totals[3] -= clock64();
	}
	#endif

	if( result == 0 ) {
		(*device.neutron_pool)[arg] = n;
		ASYNC_CALL(Fn::Neutron,arg);
	}
	
	#ifdef TIMER
	if( util::current_leader() ) {
		group.time_totals[3] += clock64();
	}
	#endif
	
	#ifdef TIMER
	if( util::current_leader() ) {
		group.time_totals[4] -= clock64();
	}
	#endif

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

		#ifdef LEVEL_CHECK
		int mlev = atomicAdd(&group.mem_level,-1);
		#endif
		
		#ifdef CACHE
		#if 0
		unsigned int idx  = util::warp_inc_scan();
		unsigned int acnt = util::active_count();
		if(idx == 0){
			printf("[%d]",acnt);
		}
		#endif
		group.cache.free(arg,thread.rand_state);
		#else
		device.neutron_pool->free(arg,thread.rand_state);
		#endif

	}

	#ifdef TIMER
	if( util::current_leader() ) {
		group.time_totals[4] += clock64();
	}
	#endif

	#ifdef TIMER
	if( util::current_leader() ) {
		group.time_totals[8] -= clock64();
	}
	#endif
}


DEF_INITIALIZE(ProgType) {

	thread.rand_state = blockDim.x * blockIdx.x + threadIdx.x;

	#ifdef CACHE
	group.cache.initialize(*device.neutron_pool);
	#endif

	#ifdef TIMER
	if( util::current_leader() ) {
		for(unsigned int i=0; i<TIMER; i++){
			group.time_totals[i] = 0;
		}
		group.time_totals[0] -= clock64();
	}
	#endif


	#ifdef LEVEL_CHECK
	if( util::current_leader() ){
		group.mem_level = 0;
		group.mem_max   = 0;
	}
	__syncwarp();
	#endif

	#ifdef TIMER
	if( util::current_leader() ) {
		group.time_totals[8] -= clock64();
	}
	#endif
}


DEF_FINALIZE(ProgType) {
	#ifdef TIMER
	if( util::current_leader() ) {
		group.time_totals[8] += clock64();
	}
	#endif

	#ifdef CACHE
	group.cache.finalize(thread.rand_state);
	#endif

	#ifdef LEVEL_CHECK
	__syncwarp();
	if( util::current_leader() ){
		atomicAdd(device.level_total,group.mem_max);
	}
	#endif

	#ifdef TIMER
	if( util::current_leader() ) {
		group.time_totals[0] += clock64();
		for(unsigned int i=0; i<TIMER; i++){
			atomicAdd(&(device.timer[i]),group.time_totals[i]);
		}
	}
	#endif

}


DEF_MAKE_WORK(ProgType) {


	#ifdef TIMER
	if( util::current_leader() ) {
		group.time_totals[8] += clock64();
	}
	#endif


	unsigned int id;


	#ifdef TIMER
	if( util::current_leader() ) {
		group.time_totals[5] -= clock64();
	}
	#endif
	
	#ifdef EVENT
	float fill_frac = QUEUE_FILL_FRACTION(Fn::Neutron);
	if( fill_frac > 0.5 ){
		return false;
	}

	iter::Iter<unsigned int> iter = device.source_id_iter->leap(2u);

	#else

	iter::Iter<unsigned int> iter = device.source_id_iter->leap(1u);

	#endif
	
	#ifdef TIMER
	if( util::current_leader() ) {
		group.time_totals[5] += clock64();
	}
	#endif


	while(iter.step(id)){

		Neutron n(id,0.0,0.0,0.0,0.0,1.0);

		#ifdef FILO
		n.next = mem::Adr<unsigned int>::null;
		#endif

		#ifdef TIMER
		if( util::current_leader() ) {
			group.time_totals[6] -= clock64();
		}
		#endif
		
		#ifdef CACHE
		unsigned int index = group.cache.alloc_index(thread.rand_state);
		#else
		unsigned int index = device.neutron_pool->alloc_index(thread.rand_state);
		#endif
		
		#ifdef TIMER
		if( util::current_leader() ) {
			group.time_totals[6] += clock64();
		}
		#endif

		#ifdef LEVEL_CHECK
		int mlev = atomicAdd(&group.mem_level,1);
		atomicMax(&group.mem_max,mlev+1);
		#endif

		if(index == mem::Adr<unsigned int>::null){
			printf("{failed to allocate}");
		}
		
		if( (index != mem::Adr<unsigned int>::null) ) {
			(*device.neutron_pool)[index] = n;

			#ifdef ALLOC_CHECK
			unsigned int old = atomicCAS(&(*device.neutron_pool)[index].checkout,0u,1u);
			if( old != 0 ){
				printf("\n{Bad alloc %d at %d}\n",old,index);
			}
			#endif

			#ifdef TIMER
			if( util::current_leader() ) {
				group.time_totals[7] -= clock64();
			}
			#endif
			
			#ifdef EVENT
			IMMEDIATE_CALL(Fn::Neutron,index);
			#else
			ASYNC_CALL   (Fn::Neutron,index);
			#endif
				
			#ifdef TIMER
			if( util::current_leader() ) {
				group.time_totals[7] += clock64();
			}
			#endif
		}
		
	}

	#ifdef TIMER
	if( util::current_leader() ) {
		group.time_totals[8] -= clock64();
	}
	#endif

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
	ProgType::Instance instance = ProgType::Instance(0x10000000,com.params);
	#else 
	ProgType::Instance instance = ProgType::Instance(0x1000000,com.params);
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
	//printf("\nIter count is %d\n",num);

	#ifdef HRM_TIME
	printf("Instance times:\n");
	instance.print_times();
	#endif

	#endif

	return 0;

}

