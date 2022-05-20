


using namespace util;


struct Step;


//template<typename PROGRAM>
struct MyDeviceState {

	iter::AtomicIter<unsigned int> *source_id_iter;

	unsigned int* counter;

	WorkBarrier<OpUnion<Step>>* barrier;

};



struct Step {

	using Type = void(*)(unsigned int);

	template<typename PROGRAM>
	__device__ static void eval(PROGRAM prog, unsigned int arg){

		if( arg > 0 ){
			atomicAdd(prog.device.counter,1);
			prog.device.barrier->atomic_append(prog,Promise<Step>(4));
		} else {
			atomicAdd(prog.device.counter,2);
			prog.device.barrier->release(prog);
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
			prog.template sync<Step>(id);
		}

		// Return whether or not the ID iterator has any IDs left
		return !prog.device.source_id_iter->done();

	}


};




using AsyncProgram = HarmonizeProgram <MyProgramSpec>;



int main(int argc, char *argv[]){


	using host::check_error;


	///////////////////////////////////////////////////////////////////////////////////////////
	// Gathering Arguments
	///////////////////////////////////////////////////////////////////////////////////////////

	// Get arguments from the command line
	cli::ArgSet args(argc,argv);

	size_t wg_count = args["wg_count"];

	unsigned int dev_idx  = args["dev_idx"] | 0;
	cudaSetDevice(dev_idx); 

	unsigned int arena_size = args["pool"] | 0x8000000;
	
	//MyDeviceState<AsyncProgram>  dev_state;
	MyDeviceState dev_state;

	host::DevBuf<iter::AtomicIter<unsigned int>> source_id_iter;
	host::DevBuf<unsigned int> counter;

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
	source_id_iter<< iter::AtomicIter<unsigned int>(0,args["num"]);
	dev_state.source_id_iter = source_id_iter;


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
		check_error();
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

	printf("\nResult is %d\n",result[0]);

	printf("\nProcessing took %f milliseconds\n",msec_total);

	

	return 0;

}


