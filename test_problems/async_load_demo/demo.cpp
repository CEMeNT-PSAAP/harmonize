


using namespace util;


struct Step;


const size_t DATA_SIZE = 32;

struct Data {
	unsigned int values[DATA_SIZE];
};

struct MyDeviceState {

	unsigned int  step_limit;
	unsigned int  input_size;
	unsigned int output_size;

	iter::AtomicIter<unsigned int> *source_id_iter;

	Data* input;
	unsigned int* output;


};

struct IterState {
	unsigned int start; 
	unsigned int step;
	unsigned int value;
};


const size_t LOOP_COUNT = 128;

struct Step {

	using Type = void(*)(IterState, Data);

	template<typename PROGRAM>
	__device__ static void eval(PROGRAM prog, IterState state, Data now){

		if(state.step < prog.device.step_limit){

			state.step++;
			for(int i=0; i<LOOP_COUNT; i++){
				for(int j=0; j<DATA_SIZE;j++){
					state.value ^= util::random_uint(now.values[i]);
				}
			}

			unsigned int  next_index = state.value + state.step;
			Data& next = prog.device.input[next_index % prog.device.input_size];
			prog. template async<Step>(state,LazyLoad<Data>(next));
			//prog. template async<Step>(state,next);

		} else {
			prog.device.output[state.start] = state.value;
		}

	}

};


struct MyProgramSpec {
	
	using OpSet =  OpUnion<Step>;
	
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

		iter::Iter<unsigned int> iter = prog.device.source_id_iter->leap(4);
		
		while(iter.step(id)){
			IterState state;
			state.step  = 0;
			state.value = 0;
			state.start = id;

			Data data;
			for(int i=0; i<DATA_SIZE;i++){
				data.values[i] = id;
			}	

			prog.template async<Step>(state,data);
		}

		return !prog.device.source_id_iter->sync_done();

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

	unsigned int arena_size = args["pool"] | 0x800000;
	
	unsigned int input_size  = args["i_size"] | 32;
	unsigned int output_size = args["o_size"] | 32;
	unsigned int step_limit  = args["limit"]  | 0;
	
	//MyDeviceState<AsyncProgram>  dev_state;
	MyDeviceState dev_state;

	dev_state. input_size =  input_size;
	dev_state.output_size = output_size;
	dev_state. step_limit =  step_limit;


	host::DevBuf<iter::AtomicIter<unsigned int>> source_id_iter;
	host::DevBuf<Data> input;
	host::DevBuf<unsigned int> output;

	Stopwatch watch;

	///////////////////////////////////////////////////////////////////////////////////////////
	// Program Setup
	///////////////////////////////////////////////////////////////////////////////////////////

	// Start watch before we start doing any significant work
	watch.start();

	std::vector<Data> inp_init;
	inp_init.resize(input_size);
	unsigned int rand_state = 0xDEADBEEF;

	for(unsigned int i=0; i <input_size;i++) {
		for(unsigned int j=0; j<DATA_SIZE; j++){
			inp_init[i].values[j] = util::random_uint(rand_state);
		}
	}

	input << inp_init;

	output.resize(output_size);
	cudaMemset(output, 0, sizeof(unsigned int)*output_size );

	dev_state. input =  input;
	dev_state.output = output;

	// Set the ID iterator to the range from zero to the number of source neutrons
	source_id_iter<< iter::AtomicIter<unsigned int>(0,output_size);
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

	output >> result;

	unsigned int sum = 0;
	for(unsigned int i=0; i<output_size; i++){
		sum += result[i];
	}
	printf("Sum is: %d\n",sum);

	printf("\nProcessing took %f milliseconds\n",msec_total);

	

	return 0;

}


