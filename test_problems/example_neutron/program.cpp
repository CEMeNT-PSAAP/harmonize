

#include "logic.cpp"

// Pull in the utilities from harmonize
using namespace util;

struct ThreadState {
	// Nothing here, to keep things simple
};


struct GroupState {
	// Nothing here, to keep things simple
};


// Define our program state definition
typedef ProgramStateDef<DeviceState,GroupState,ThreadState> ProgState;
//                        ^
//                        Our device state, as defined in logic.cpp

// Declare the set of operations
enum class Fn { Neutron };

// Declare the argument types corresponding to the operations
DEF_PROMISE_TYPE(Fn::Neutron, Neutron);


// Define our program type
#ifdef EVENT
typedef  EventProgram     < PromiseUnion<Fn::Neutron>, ProgState > ProgType;
#else
typedef  HarmonizeProgram < PromiseUnion<Fn::Neutron>, ProgState, unsigned int,  8, 8191, 8191 > ProgType;
#endif




// The single operation for processing neutrons
DEF_ASYNC_FN(ProgType, Fn::Neutron, arg) {

	Neutron n = arg;

	// Keep performing steps until the horizon is reached
	int result = 0;
	for(int i=0; i < device.horizon; i++){
		result = step_neutron(device,n);
		if( result != 0 ){
			break;
		}
	}
	
	// If result is positive, generate that number of neutrons from fission
	for(int i=0; i<result; i++){
		Neutron new_neutron = n.child();
		ASYNC_CALL(Fn::Neutron,new_neutron);
	}

	// If result is zero, re-queue neutron for further processing
	if( result == 0 ) {
		ASYNC_CALL(Fn::Neutron,n);
	}

	// If result is negative, do nothing

}


DEF_INITIALIZE(ProgType) {

	// Nothing here, to keep things simple

}


DEF_FINALIZE(ProgType) {

	// Nothing here, to keep things simple

}


// Function for queueing new neutrons when work is running low
DEF_MAKE_WORK(ProgType) {

	unsigned int id;
	
	#ifdef EVENT
	// If buffer is greater than 50% full, stop generating work to prevent over-filling
	// due to fission
	float fill_frac = QUEUE_FILL_FRACTION(Fn::Neutron);
	if( fill_frac > 0.5 ){
		return false;
	}
	#endif

	// How many neutron IDs we want to claim from the iterator each time the make_work function is called
	unsigned int iter_step_length = 1u;

	// Claim the next set of IDs
	iter::Iter<unsigned int> iter = device.source_id_iter->leap(iter_step_length);
	
	// Keep processing new neutrons until the iterator runs out
	while(iter.step(id)){
		Neutron n(id,0.0,0.0,0.0,0.0,1.0);
		IMMEDIATE_CALL(Fn::Neutron,n);
	}

	// Return whether or not the ID iterator has any IDs left
	return !device.source_id_iter->done();

}



int main(int argc, char *argv[]){

	using host::check_error;


	///////////////////////////////////////////////////////////////////////////////////////////
	// Gathering Arguments
	///////////////////////////////////////////////////////////////////////////////////////////

	// Get arguments from the command line
	cli::ArgSet args(argc,argv);

	// The number of work groups to use
	unsigned int wg_count = args["wg_count"];

	// The device index to use (zero is default)
	unsigned int dev_idx  = args["dev_idx"] | 0;
	cudaSetDevice(dev_idx); 

	// Whether or not to show a graph on the command line
	bool	    show = args["show"];

	// Whether or not to output census data as CSV to standard out
	bool	    csv  = args["csv"];
	
	// The size of the arena/io buffer used by the async/event-based program
	unsigned int arena_size = args["pool"] | 0x8000000;
	
	DeviceState  dev_state;

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
	dev_state.source_count  = args["num"]  | 1000u;;

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

	// Initialize instance
	#ifdef EVENT
	ProgType::Instance instance = ProgType::Instance(arena_size,dev_state);
	#else 
	ProgType::Instance instance = ProgType::Instance(arena_size/32u,dev_state);
	#endif

	// Sync for safety and report any errors
	cudaDeviceSynchronize();
	check_error();
	
	///////////////////////////////////////////////////////////////////////////////////////////
	// Execution
	///////////////////////////////////////////////////////////////////////////////////////////
	
	#ifdef EVENT

	// While the instance has not yet completed, execute, sync, and report any errors
	do {
		// Give the number of work groups and the size of the chunks pulled from
		// the io buffer
		exec<ProgType>(instance,wg_count,1);
		cudaDeviceSynchronize();
		check_error();
	} while ( ! instance.complete() );

	#else	

	init<ProgType>(instance,wg_count);
	cudaDeviceSynchronize();
	check_error();
	int num = 0;
	
	// While the instance has not yet completed, execute, sync, and report any errors
	do {
		// Give the number of work groups used and the number of iterations
		// to perform before halting early, to prevent GPU timeouts
		exec<ProgType>(instance,wg_count,0x10000);
		cudaDeviceSynchronize();
		check_error();
		num++;
	} while(! instance.complete() );

	#endif


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

