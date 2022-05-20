

#include "logic.cpp"

// Pull in the utilities from harmonize
using namespace util;



// The single operation for processing neutrons
struct Step {

	using Type = void(*)(Neutron);

	template<typename PROGRAM>
	__device__ static void eval(PROGRAM prog, Neutron arg){
		Neutron n = arg;

		// Keep performing steps until the horizon is reached
		int result = 0;
		for(int i=0; i < prog.device.horizon; i++){
			result = step_neutron(prog.device,n);
			if( result != 0 ){
				break;
			}
		}
		
		// If result is positive, generate that number of neutrons from fission
		for(int i=0; i<result; i++){
			Neutron new_neutron = n.child();
			prog. template async<Step>(new_neutron);
		}

		// If result is zero, re-queue neutron for further processing
		if( result == 0 ) {
			prog. template async<Step>(n);
		}

		// If result is negative, do nothing
		

	}

};


// Define a specification for our program 
struct MyProgramSpec {

	// Define a type called OpSet (specializing OpUnion) to define what operations
	// the program uses. Since this is a specialization of OpUnion, this should be done
	// as a typedef or similar construct. The default type assumed is OpUnion<>, which
	// contains no operations.
	typedef OpUnion<Step> OpSet;



	// Define a type called DeviceState to define what state is made available on a
	// per-device basis. This can be done by a typedef or by manually declaring a type.
	// If no definition is provided, the memberless, zero-size VoidState will be used.
	//
	// The device state, itself, is an immutable struct, but can contain references
	// and pointers to non-const data.
	typedef MyDeviceState DeviceState;
	
	// Same idea as DeviceState, but it is a state tracked on a per-group basis, and
	// the state provided will be directly mutable.
	struct GroupState {
		// Nothing here, to keep things simple
	};

	// Same idea as GroupState, but it is a state tracked on a per-thread basis.
	struct ThreadState {
		// Nothing here, to keep things simple
	};
	

	// Defines what integer type is used to represent the address of intermediate
	// data stored in global memory. If unset, the default type assumed is uint32_t.
	// Using a larger representation can accomodate for more data. Using  smaller
	// representation can make some aspects of managing intermediate data more efficient.
	typedef unsigned int AdrType;


	// The size of the local storage used by the asynchronous scheduler.
	// The current default, if undeclared, is 16
	static const size_t STASH_SIZE = 8;

	// The number of queues used by the asynchronous method to store valid work in main memory.
	// The current default, if undecleared, is 32
	static const size_t FRAME_SIZE = 8191;
	
	// The number of queues used by the asynchronous method to store unused intermediate data storage.
	// The current default, if undecleared, is 32
	static const size_t POOL_SIZE  = 8191;



	// A function called by each work group at the start of each execution pass, before
	// any operations are processed.
	template<typename PROGRAM>
	__device__ static void initialize(PROGRAM prog) {}

	// A function called by each work group at the end of each execution pass, after the
	// final operation evaluations for the pass have occured for that particular work group.
	template<typename PROGRAM>
	__device__ static void finalize(PROGRAM prog) {}



	// A function called by each work group if it detects scarcity of work. This function is called
	// every time such a detection occurs during an execution pass until the first time a false
	// value is returned for that work group during that execution pass. This function may be 
	// subsequently called in different execution passes.
	template<typename PROGRAM>
	__device__ static bool make_work(PROGRAM prog) {

		unsigned int id;

		if( ! prog.device.is_async ) {
			// If buffer is greater than 50% full, stop generating work to prevent over-filling
			// due to fission
			float fill_frac = prog.template queue_fill_fraction<Step>();
			if( fill_frac > 0.5 ){
				return false;
			}
		}

		// How many neutron IDs we want to claim from the iterator each time the make_work function is called
		unsigned int iter_step_length = 1u;

		// Claim the next set of IDs
		iter::Iter<unsigned int> iter = prog.device.source_id_iter->leap(iter_step_length);
		
		// Keep processing new neutrons until the iterator runs out
		while(iter.step(id)){
			Neutron n(id,0.0,0.0,0.0,0.0,1.0);
			prog.template sync<Step>(n);
		}

		// Return whether or not the ID iterator has any IDs left
		return !prog.device.source_id_iter->done();

	}


};




// Define our program types
using  SyncProgram = EventProgram     <MyProgramSpec>;
using AsyncProgram = HarmonizeProgram <MyProgramSpec>;




struct VoidOp {

	using Type = void(*)(void);

	template<typename PROGRAM>
	__device__ static void eval(void){}

};


int main(int argc, char *argv[]){


	using host::check_error;


	///////////////////////////////////////////////////////////////////////////////////////////
	// Gathering Arguments
	///////////////////////////////////////////////////////////////////////////////////////////

	// Get arguments from the command line
	cli::ArgSet args(argc,argv);

	// The number of work groups to use
	size_t wg_count = args["wg_count"];

	// The device index to use (zero is default)
	unsigned int dev_idx  = args["dev_idx"] | 0;
	cudaSetDevice(dev_idx); 

	// Whether or not to show a graph on the command line
	bool	    show = args["show"];

	// Whether or not to output census data as CSV to standard out
	bool	    csv  = args["csv"];
	
	// The size of the arena/io buffer used by the async/event-based program
	unsigned int arena_size = args["pool"] | 0x8000000;
	
	MyDeviceState  dev_state;

	// The source ID iterator
	host::DevBuf<iter::AtomicIter<unsigned int>> source_id_iter;

	// The census tallies, which are indexed by the neutron's x position
	// during census
	host::DevBuf<float> halted;

	// A timer to judge program performance
	Stopwatch watch;

	// How many times a neutron is stepped before being re-queued 
	dev_state.horizon    = args["hrzn"] | 1u;
	
	// How many neutrons the source generates
	dev_state.source_count  = args["num"]  | 1000u;

	// How long the simulation runs
	dev_state.time_limit = args["time"] | 1.0f;

	// The resolution of the tallies
	dev_state.div_width  = args["res"]  | 1.0f;

	// The bounds of the tallies in space
	dev_state.pos_limit  = args["size"] | 1.0f;
	
	// Used to determine size of tally array
	dev_state.div_count  = dev_state.pos_limit/dev_state.div_width;

	// fission, capturing, and scattering cross section
	dev_state.fission_x  = args["fx"];
	dev_state.capture_x  = args["cx"];
	dev_state.scatter_x  = args["sx"];

	// Total of all cross sections
	dev_state.combine_x  = dev_state.fission_x + dev_state.capture_x + dev_state.scatter_x;

	// How many neutrons are created by each fission event
	dev_state.fiss_mult  = args["mult"] | 2;

	// Whether or not implicit capture is used
	dev_state.implicit_capture  = args["imp_cap"];
	// The weight limit used for implicit capture (if used)
	dev_state.weight_limit      = args["wlim"] | 0.0001f;

	// Whether or not async scheduling is used
	dev_state.is_async  = args["async"];

	///////////////////////////////////////////////////////////////////////////////////////////
	// Program Setup
	///////////////////////////////////////////////////////////////////////////////////////////

	// Start watch before we start doing any significant work
	watch.start();

	// Set tally array size and initialize with zeros
	int elem_count = dev_state.div_count*2;
	halted.resize(elem_count);
	dev_state.halted = halted;
	cudaMemset( halted, 0, sizeof(float) * elem_count );

	// Set the ID iterator to the range from zero to the number of source neutrons
	source_id_iter<< iter::AtomicIter<unsigned int>(0,args["num"]);
	dev_state.source_id_iter = source_id_iter;


	// Synchronize for safety and report any errors
	cudaDeviceSynchronize();
	check_error();


	///////////////////////////////////////////////////////////////////////////////////////////
	// Execution
	///////////////////////////////////////////////////////////////////////////////////////////
	
	if( !dev_state.is_async ){

		SyncProgram::Instance instance(arena_size,dev_state);

		// Sync for safety and report any errors
		cudaDeviceSynchronize();
		check_error();
		
		// While the instance has not yet completed, execute, sync, and report any errors
		do {
			// Give the number of work groups and the size of the chunks pulled from
			// the io buffer
			exec<SyncProgram>(instance,wg_count,1);
			cudaDeviceSynchronize();
			check_error();
		} while ( ! instance.complete() );

	} else {
		
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

	}


	
	///////////////////////////////////////////////////////////////////////////////////////////
	// Program Wrap-up
	///////////////////////////////////////////////////////////////////////////////////////////


	// Stop timer and get the intervening milliseconds
	watch.stop();
	float msec_total = watch.ms_duration();
	
	// Declare container for result data
	std::vector<float> result;

	// If either show or csv is true, then gather the result data
	float y_min, y_max;
	float sum = 0;
	int real_sum = 0;
	if( show || csv ){

		// Get data from the GPU
		halted >> result;

		// Iterate through the data an scale it by the inverse of the number
		// of source particles
		for(unsigned int i=0; i<elem_count; i++){
			real_sum += result[i];
			result[i] /= (float) dev_state.source_count;
			sum += result[i];
			result[i] /= (float) dev_state.div_width;
			if( i == 0 ){
				y_min = result[i];
				y_max = result[i];
			}
			y_min = (result[i] < y_min) ? result[i] : y_min;
			y_max = (result[i] > y_max) ? result[i] : y_max;
		}

	}


	// Print out data as csv, if necessary
	if( csv ){
		for(unsigned int i=0; i<elem_count; i++){
			if( i == 0 ){
				printf("%f",result[i]);
			} else {
				printf(",%f",result[i]);
			}
		}
		printf("\n");
	
		return 0;
	}

	// If not, and show is true, feed data into cli graphing tool	
	if( show ){

		util::cli::GraphShape shape;
		shape.y_min  = y_min;
		shape.y_max  = y_max;
		shape.x_min  = -dev_state.pos_limit;
		shape.x_max  =  dev_state.pos_limit;
		shape.width  = 100;
		shape.height = 16;

		util::cli::cli_graph(result.data(),elem_count,shape,util::cli::Block2x2Fill);

	}

	printf("\nProcessing took %f milliseconds\n",msec_total);

	

	return 0;

}


