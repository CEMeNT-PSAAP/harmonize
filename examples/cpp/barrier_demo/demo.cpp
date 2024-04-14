


using namespace util;


struct Step;


struct MyDeviceState {

	size_t dep_count;
	size_t non_dep_count;
	size_t barrier_count;

	iter::AtomicIter<unsigned int> *source_id_iter;

	unsigned int* counter;

	RemappingBarrier<OpUnion<Step>>* barrier;

};



struct Step {

	using Type = void(*)(unsigned int);

	template<typename PROGRAM>
	__device__ static void eval(PROGRAM prog, unsigned int arg){

		//printf("Arg is %d\n",arg);
		atomicAdd(prog.device.counter,1);
		if ( arg < prog.device.dep_count ){
			size_t barrier_idx = arg % prog.device.barrier_count;
			RemappingBarrier<OpUnion<Step>>& barrier = prog.device.barrier[barrier_idx];
			barrier.await(prog,Promise<Step>(prog.device.dep_count));
			barrier.add_semaphore(prog,(unsigned int)-1);
		}

	}

};


struct MyProgramSpec {
	
	using OpSet =  OpUnion<Step>;
	
	//template<typename T> using DeviceState = MyDeviceState<T>;
	using DeviceState = MyDeviceState;
	
	using AdrType = unsigned int;
	
	template<typename PROGRAM>
	struct GroupState {};
	
	template<typename PROGRAM>
	struct ThreadState {};
	
	
	static const size_t STASH_SIZE = 8;
	static const size_t FRAME_SIZE = 8191;
	static const size_t POOL_SIZE  = 8191;
	
	template<typename PROGRAM>
	__device__ static void initialize(PROGRAM prog) {}

	template<typename PROGRAM>
	__device__ static void finalize(PROGRAM prog) {}

	template<typename PROGRAM>
	__device__ static bool make_work(PROGRAM prog) {

		unsigned int id;

		iter::Iter<unsigned int> iter = prog.device.source_id_iter->leap(1);
		
		while(iter.step(id)){
			//printf("Got id %d as thread %d\n",id,threadIdx.x);
			prog.template sync<Step>(id);
		}

		
		// Return whether or not the ID iterator has any IDs left
		return !prog.device.source_id_iter->sync_done();

	}


};




using AsyncProgram = HarmonizeProgram <MyProgramSpec>;



int main(int argc, char *argv[]){


	using host::check_error;


	float total = 0;
	int samp_count = 32;

	for(int i=0; i<samp_count; i++){
	///////////////////////////////////////////////////////////////////////////////////////////
	// Gathering Arguments
	///////////////////////////////////////////////////////////////////////////////////////////

	// Get arguments from the command line
	cli::ArgSet args(argc,argv);

	size_t wg_count = args["wg_count"];

	unsigned int dev_idx  = args["dev_idx"] | 0;
	cudaSetDevice(dev_idx); 

	unsigned int arena_size = args["pool"] | 0x800000;
	
	//MyDeviceState<AsyncProgram>  dev_state;
	MyDeviceState dev_state;

	dev_state.dep_count = args["dep"];
	dev_state.barrier_count = args["bar"];
	dev_state.non_dep_count = args["ndep"];

	host::DevBuf<iter::AtomicIter<unsigned int>> source_id_iter;
	host::DevBuf<unsigned int> counter;
	host::DevBuf<RemappingBarrier<OpUnion<Step>>> barrier;

	Stopwatch watch;

	///////////////////////////////////////////////////////////////////////////////////////////
	// Program Setup
	///////////////////////////////////////////////////////////////////////////////////////////

	// Start watch before we start doing any significant work
	watch.start();

	counter.resize(1);
	dev_state.counter = counter;
	cudaMemset( counter, 0, sizeof(unsigned int) );

	// Set the ID iterator to the range from zero to the number of source neutrons
	source_id_iter<< iter::AtomicIter<unsigned int>(0,dev_state.dep_count+dev_state.non_dep_count);
	dev_state.source_id_iter = source_id_iter;
	
	std::vector<RemappingBarrier<OpUnion<Step>>> barrier_init;
	for(size_t i=0; i < dev_state.barrier_count; i++){
		size_t wait_count = (dev_state.dep_count / dev_state.barrier_count);
		if( (dev_state.dep_count % dev_state.barrier_count) > i ){
			wait_count++;
		}
		barrier_init.push_back(RemappingBarrier<OpUnion<Step>>::blank(wait_count));
	}
	barrier << barrier_init;
	dev_state.barrier = barrier;

	// Synchronize for safety and report any errors
	cudaDeviceSynchronize();
	check_error();


	///////////////////////////////////////////////////////////////////////////////////////////
	// Execution
	///////////////////////////////////////////////////////////////////////////////////////////
	
		
	AsyncProgram::Instance instance(arena_size/32u,dev_state);

	// Sync for safety and report any errors
	cudaDeviceSynchronize();
	check_error();
	
	init<AsyncProgram>(instance,wg_count);
	cudaDeviceSynchronize();
	check_error();
	int num = 0;
	
	// While the instance has not yet completed, execute, sync, and report any errors
	do {
		// Give the number of work groups used and the number of iterations
		// to perform before halting early, to prevent GPU timeouts
		exec<AsyncProgram>(instance,wg_count,0x1000000);
		cudaDeviceSynchronize();
		if( check_error() ) {
			break;
		}
		num++;
	} while(! instance.complete() );



	
	///////////////////////////////////////////////////////////////////////////////////////////
	// Program Wrap-up
	///////////////////////////////////////////////////////////////////////////////////////////


	// Stop timer and get the intervening milliseconds
	watch.stop();
	float msec_total = watch.ms_duration();
	
	// Declare container for result data
	std::vector<unsigned int> result;

	counter >> result;

	//printf("\nResult is %d and should be %d\n",result[0],dev_state.dep_count*2+dev_state.non_dep_count);

	//printf("\nProcessing took %f milliseconds\n",msec_total);
	total+= msec_total;
	
	}


	printf("%f",total/samp_count);

	

	return 0;

}


