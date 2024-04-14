
using namespace util;



// Define a 'collaz' struct to hold the arguments for the even and odd async functions
struct Collaz{ unsigned int step; unsigned int original; unsigned long long int val; };

struct Even;
struct Odd;


struct Even {

	using Type = void(*)(unsigned int step, unsigned int original, unsigned long long int val);


	template<typename PROGRAM>
	__device__ static void eval(PROGRAM prog, unsigned int step, unsigned int original, unsigned long long int val) {

		if( val <= 1 ){
			prog.device.output[original] = step;
			return;
		}

		step += 1;
		val  /= 2;

		if( (val%2) == 0 ){
			prog.template async<Even>(step,original,val);
		} else {
			prog.template async< Odd>(step,original,val);
		}

	}

};



struct Odd{

	using Type = void(*)(unsigned int step, unsigned int original, unsigned long long int val);

	template<typename PROGRAM>
	__device__ static void eval(PROGRAM prog, unsigned int step, unsigned int original, unsigned long long int val) {

		if( val <= 1 ){
			prog.device.output[original] = step;
			return;
		}

		step += 1;
		val  *= 3;
		val  += 1;

		if( (val%2) == 0 ){
			prog.template async<Even>(step,original,val);
		} else {
			prog.template async< Odd>(step,original,val);
		}

	}

};

// The state that will be stored per program instance and accessible by all work groups
struct MyDeviceState{
	unsigned int start;
	unsigned int limit;
	unsigned int *output;
	iter::AtomicIter<unsigned int>* iterator;
};



struct MyProgramSpec {


	typedef OpUnion<Even,Odd>       OpSet;
	typedef     MyDeviceState DeviceState;

	static const size_t STASH_SIZE =   16;
	static const size_t FRAME_SIZE = 8191;
	static const size_t  POOL_SIZE = 8191;




	/*
	// Defines the initialization function for programs of type 'ProgType'. This function is called by
	// all threads in all work groups just prior to beginning normal execution. States are accessible
	// through the 'device', 'group', and 'thread' variables, just like when defining async functions.
	//
	// Here, we initialize the work iterator to iterate over a range of integers unique to the work
	// group, distributing the set of integers to process more or less evenly across the set of
	// work groups.
	*/
	template<typename PROGRAM>
	__device__ static void initialize(PROGRAM prog){

	}

	/*
	// Defines a function for programs of type 'ProgType' which is called by all threads in all work
	// groups after it is determined that there are no more promises to be processed. To be clear:
	// you should not perform any async calls in here and expect them to be always evaluated, since
	// the program is wrapping up and there is a good chance that no work groups will notice the
	// promises before exiting.
	//
	// Because promises are persistant across execution calls, if you want to queue work for the next
	// execution call, you can check if the current executing work group is the final one to call the
	// finalize function and queue work then. This will guarantee that the queued work will only be
	// evaluated in the next exec call.
	*/
	template<typename PROGRAM>
	__device__ static void finalize(PROGRAM prog){


	}



	/*
	// Defines the work making function for programs of type 'ProgType'. This function is called by
	// work groups whenever they notice that they are running out of work to perform. To indicate
	// that there is still more work to perform, return 'true'. To indicate that there is no more
	// work left for the work group to make, return 'false', at which point, the work group will no
	// longer call this function for the remainder of the execution run.
	*/
	template<typename PROGRAM>
	__device__ static bool make_work(PROGRAM prog){

		unsigned int iter_step_length = 1u;

		iter::Iter<unsigned int> iter = prog.device.iterator->leap(iter_step_length);

		unsigned int index;
		while(iter.step(index)){
			if( (index % 2) == 0 ){
				prog.template async<Even>(0,index,index);
			} else {
				prog.template async< Odd>(0,index,index);
			}
		}

		return ! prog.device.iterator->done();

	}



};


using ProgType = AsyncProgram < MyProgramSpec >;




// A function to double-check our work
unsigned int checker(unsigned long long int val){
	unsigned int res = 0;
	while(val > 1){
		res++;
		if( (val%2) == 0){
			val /=2;
		} else {
			val *=3;
			val +=1;
		}
	}
	return res;
}



int main(int argc, char* argv[]){

	cli::ArgSet args(argc,argv);


	unsigned int wg_count    = args["wg_count"];
	unsigned int cycle_count = args["cycle_count"] | 0x100000u;

	Stopwatch watch;

	watch.start();

	// Declare a device state and initialize it with the information  it requires:
	MyDeviceState ds;

	// Define the range of values to process through the collaz iteration
	ds.start  = 0;
	ds.limit  = args["limit"];

	// Define a device-side buffer of type 'unsigned int' with size 'ds.limit'
	host::DevBuf<unsigned int> dev_out = host::DevBuf<unsigned int>((size_t)ds.limit);
	// Assign the address of the device-side buffer to the device state so that the program
	// can know where to put its output.
	ds.output = dev_out;

	iter::AtomicIter<unsigned int> host_iter(0,ds.limit);
	host::DevBuf<iter::AtomicIter<unsigned int>> iterator;
	iterator << host_iter;
	ds.iterator = iterator;


	// Declare and instance of type 'ProgType' with an arena size of 2^(20) with a device state
	// initialized to the value of our declared device state struct. The arena size of a
	// program determines how much extra space it has to store work if it cannot store
	// everything inside shared memory. If you are *certain* that no work will spill into
	// main memory, you may get some performance benefits by seting the arena size to zero.
	ProgType::Instance instance = ProgType::Instance(0x10000,ds);
	cudaDeviceSynchronize();
	host::check_error();

	// Initialize the instance using 32 work groups
	init<ProgType>(instance,32);
	cudaDeviceSynchronize();
	host::check_error();

	// Execute the instance using 240 work groups, with each work group performing up to
	// 65536 promise executions per thread before halting. If all promises are exhausted
	// before this, the program exits early.
	exec<ProgType>(instance,wg_count,cycle_count);
	cudaDeviceSynchronize();
	host::check_error();

	watch.stop();

	float msec = watch.ms_duration();

	// Retrieve the data from the device-side buffer into a host-side vector.
	std::vector<unsigned int> host_out(ds.limit);
	dev_out >> host_out;
	host::check_error();


	// Double-check that the program has accurately evaluated the iterations assigned to it.
	for(unsigned int i=0; i<ds.limit; i++){
		if( host_out[i] != checker(i) ){
			printf("Bad output for input %d. Output is %d when it should be %d\n",i,host_out[i],checker(i));
		}
	}

	printf("Runtime: %f\n",msec);

	return 0;

}

