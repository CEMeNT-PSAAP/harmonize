

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
typedef mem::SimpleMemCache<PoolType,16>      CacheType;
#endif


struct MyThreadState {

	unsigned int rand_state;

};


struct MyGroupState {

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



struct Step {


	using Type = void(*)(Neutron); 


	template<typename PROGRAM>
	__device__ static void eval(PROGRAM prog, Neutron arg) {

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
		for(int i=0; i < prog.device.horizon; i++){
			result = step_neutron(prog.device,n);
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
			int mlev = atomicAdd(&prog.group.mem_level,1);
			atomicMax(&prog.group.mem_max,mlev+1);
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
				prog.template async<Step>(new_neutron);
			#endif


		}

		end_clock(2);


		beg_clock(3);

		if( result == 0 ) {
			#ifdef BY_REF
			(*device.neutron_pool)[arg] = n;
			prog.template async<Step>(arg);
			#else
			prog.template async<Step>(n);
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
			int mlev = atomicAdd(&prog.group.mem_level,-1);
		}
		#endif

		end_clock(4);

		beg_clock(8);


	}

};





struct MyProgramSpec {


	static const size_t STASH_SIZE =    8;
	static const size_t FRAME_SIZE = 8191;
	static const size_t  POOL_SIZE = 8191;

	typedef OpUnion<Step> OpSet;

	typedef     SimParams DeviceState;
	typedef  MyGroupState  GroupState;
	typedef MyThreadState ThreadState;

	template<typename PROGRAM>
	__device__ static void initialize(PROGRAM prog) {
		prog.thread.rand_state = blockDim.x * blockIdx.x + threadIdx.x;

		#ifdef BY_REF
		#ifdef CACHE
		prog.group.cache.initialize(*device.neutron_pool);
		#endif
		#endif

		#ifdef TIMER
		if( util::current_leader() ) {
			for(unsigned int i=0; i<TIMER; i++){
				prog.group.time_totals[i] = 0;
			}
		}
		beg_clock(0);
		#endif

		#ifdef LEVEL_CHECK
		if( util::current_leader() ){
			prog.group.mem_level = 0;
			prog.group.mem_max   = 0;
		}
		__syncwarp();
		#endif

		beg_clock(8);
	}




	template<typename PROGRAM>
	__device__ static void finalize(PROGRAM prog) {
		end_clock(8);

		#ifdef BY_REF
		#ifdef CACHE
		group.cache.finalize(prog.thread.rand_state);
		#endif
		#endif

		#ifdef LEVEL_CHECK
		__syncwarp();
		if( util::current_leader() ){
			atomicAdd(prog.device.level_total,prog.group.mem_max);
		}
		#endif

		#ifdef TIMER
		end_clock(0);
		if( util::current_leader() ) {
			for(unsigned int i=0; i<TIMER; i++){
				atomicAdd(&(prog.device.timer[i]),prog.group.time_totals[i]);
			}
		}
		#endif
	}


	template<typename PROGRAM>
	__device__ static bool make_work (PROGRAM prog) {

		end_clock(8);


		unsigned int id;

		beg_clock(5);

		if( ! prog.device.is_async ) {		
			float fill_frac = prog. template load_fraction<Step>();
			if( fill_frac > 0.5 ){
				return false;
			}
		}

		iter::Iter<unsigned int> iter = prog.device.source_id_iter->leap(1u);


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
			int mlev = atomicAdd(&prog.group.mem_level,1);
			atomicMax(&prog.group.mem_max,mlev+1);
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
			prog.template sync<Step>(n);
			//ASYNC_CALL(Fn::Neutron,n);
			#endif
			
			
			end_clock(12);
			
		}
		end_clock(9);

		beg_clock(8);

		return !prog.device.source_id_iter->done();

	}




};




typedef EventProgram    <MyProgramSpec> SyncProg;
typedef HarmonizeProgram<MyProgramSpec> AsyncProg;



int main(int argc, char *argv[]){

	using host::check_error;

	cli::ArgSet args(argc,argv);

	unsigned int dup    = args["dup"] | 1;

	for(unsigned int i=0; i<dup; i++){

		{
			unsigned int wg_count = args["wg_count"];

			unsigned int dev_idx  = args["dev_idx"] | 0;
			cudaSetDevice(dev_idx); 

			CommonContext com(args);

			cudaDeviceSynchronize();
				
			check_error();

			//#ifndef BY_REF
			unsigned int arena_size = args["pool"] | 0x8000000;
			//#endif

			#ifdef LEVEL_CHECK
			int high_level = 0;
			#endif

			if( com.params.is_async ){

				AsyncProg::Instance instance(arena_size/32u,com.params);
				cudaDeviceSynchronize();
				check_error();

				init<AsyncProg>(instance,wg_count);
				cudaDeviceSynchronize();
				check_error();
				int num = 0;
				do {
					exec<AsyncProg>(instance,wg_count,0x10000);//0x800);
					cudaDeviceSynchronize();
					if( check_error() ){
						return 1;
					}
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

			} else {

				SyncProg ::Instance instance(arena_size,com.params);
				cudaDeviceSynchronize();
				check_error();

				do {
					exec<SyncProg>(instance,wg_count,1);
					cudaDeviceSynchronize();
					if( check_error() ){
						return 1;
					}
					#ifdef LEVEL_CHECK
					int level;
					com.level_total >> level;
					high_level = (level > high_level) ? level : high_level;
					com.level_total << 0;
					#endif
					//printf("\n---\n");
				} while ( ! instance.complete() );


			}

			#ifdef LEVEL_CHECK
			printf("%d",high_level);
			#endif
		}

		if( i != (dup-1) ){
			printf(";");
		}
	}

	return 0;

}

