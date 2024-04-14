

#include "neut_common.cpp"


#ifdef TIMER
	#define beg_clock(idx) if(util::current_leader()) { group.time_totals[idx] -= clock64(); }
	#define end_clock(idx) if(util::current_leader()) { group.time_totals[idx] += clock64(); }
#else
	#define beg_clock(idx) ;
	#define end_clock(idx) ;
#endif




using namespace util;


struct MyThreadState {

	unsigned int rand_state;

};


struct MyGroupState {

	#ifdef TIMER
	unsigned long long int time_totals[TIMER];
	#endif

};



struct Step {


	using Type = void(*)(Neutron); 


	template<typename PROGRAM>
	__device__ static void eval(PROGRAM prog, Neutron arg) {

		end_clock(8);
		
		Neutron n = arg;
		n.step_count += 1;

		beg_clock(1);

		StepResult result;
		for(int i=0; i < prog.device.horizon; i++){
			result = step_neutron(prog.device,n);
			if( result.type != StepResultType::SCATTER ){
				break;
			}
		}
		
		end_clock(1);

		beg_clock(2);

		/*
		if( result.census >= 98 ) {
			if ( result.type == StepResultType::LOSS ) {
				printf("<<%d:LOSS>>\n",result.census);
			} else if ( result.type == StepResultType::FISSION ) {
				printf("<<%d:FISSION>>\n",result.census);
			} else if ( result.type == StepResultType::CAPTURE ) {
				printf("<<%d:CAPTURE>>\n",result.census);
			} else if ( result.type == StepResultType::SCATTER ) {
				printf("<<%d:SCATTER>>\n",result.census);
			} else if ( result.type == StepResultType::CENSUS ) {
				printf("<<%d:CENSUS>>\n",result.census);
			}
		}
		*/

		if( result.type == StepResultType::FISSION ){
			for(int i=0; i<result.value; i++){
				Neutron new_neutron = n.child();
				//prog.device.pcb[0]->add_prior();
				prog.device.pcb[result.census%3]->add_prior();
				prog.template async<Step>(new_neutron);
			}
		}

		end_clock(2);


		beg_clock(3);


		if( result.type == StepResultType::SCATTER ) {
			prog.template async<Step>(n);
		} else if ( result.type == StepResultType::CENSUS ) {
			if( result.value+1 < prog.device.census_count ){
				//prog.device.pcb[0]->give(n);
				prog.device.pcb[result.census%3]->give(n);
				//printf("<%d>\n",result.census);
			} else {
				//printf("<!%d>\n",result.census);
			}
		} else {
			prog.device.pcb[result.census%3]->resolve_prior();
			//prog.device.pcb[0]->resolve_prior();
			/*
			*/
		}

		end_clock(3);

		beg_clock(4);

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

		#ifdef TIMER
		if( util::current_leader() ) {
			for(unsigned int i=0; i<TIMER; i++){
				prog.group.time_totals[i] = 0;
			}
		}
		beg_clock(0);
		#endif

		beg_clock(8);
	}




	template<typename PROGRAM>
	__device__ static void finalize(PROGRAM prog) {
		end_clock(8);

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
			beg_clock(11);
			Neutron n(id,0.0,0.0,0.0,0.0,1.0);
			n.cohort = id % prog.device.cohort_count;
			n.step_count = 0;
			end_clock(11);

			beg_clock(12);
			prog.template sync<Step>(n);
			end_clock(12);
		}
		end_clock(9);

		beg_clock(8);



		for(int i=0; i<prog.device.cohort_count; i++){
			Neutron neut;
			bool success = false;
			for(int j=0; j<3; j++){
				if(prog.device.pcb[i*3+j]->take(neut)){
					success = true;
					break;
				}
			}
			if( success ){
				//printf("(Q)");
				prog.template sync<Step>(neut);
			}
		}

		bool pcb_work = false;
		for(int i=0; i<prog.device.cohort_count; i++){
			for(int j=0; j<3; j++){
				pcb_work = pcb_work || (!prog.device.pcb[i*3+j]->carry_iter.done());
			}
		}


		return  (!prog.device.source_id_iter->done()) || pcb_work;

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

			unsigned int arena_size = args["pool"] | 0x8000000;


			if( com.params.is_async ){

				AsyncProg::Instance instance(arena_size/32u,com.params);
				cudaDeviceSynchronize();
				check_error();

				init<AsyncProg>(instance,wg_count);
				cudaDeviceSynchronize();
				check_error();
				int num = 0;
				for(int i=0; i<com.params.census_count;i++){
					printf("{OUTER}\n");
					do {
						exec<AsyncProg>(instance,wg_count,0x10000);//0x800);
						cudaDeviceSynchronize();
						if( check_error() ){
							return 1;
						}
						//ProgType::runtime_overview(instance);
						num++;
					} while(! instance.complete() );
				}
				//printf("\nIter count is %d\n",num);
				//printf("Completed\n");


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
				} while ( ! instance.complete() );


			}

		}

		if( i != (dup-1) ){
			printf(";");
		}
	}

	return 0;

}

