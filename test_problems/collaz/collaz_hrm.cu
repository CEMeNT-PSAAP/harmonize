
// Don't forget to include harmonize
#include "../../harmonize_2.cu"


// The state that will be stored per program instance and accessible by all work groups
struct GlobalState{
	unsigned int start;
	unsigned int limit;
	unsigned int *output;
};

// The state that will be stored per work group
struct GroupState {
	util::GroupWorkIter<unsigned int> iterator;
};

/*
// Declare a certain Global+Group+Thread state tuple and name it 'ProgState' for convenience.
// We do not need a per-thread state and so fill it with the zero-size type 'VoidState'.
*/
typedef ProgramStateDef<GlobalState,GroupState,VoidState> ProgState;

// Name the identifiers for the async functions we plan to use
enum class Fn { Even, Odd };

// Define a 'collaz' struct to hold the arguments for the even and odd async functions
struct collaz{ unsigned int step; unsigned int original; unsigned int val; };

// Declare that our even and odd async functions accept an argument of type 'collaz'
DEF_PROMISE_TYPE(Fn::Even, collaz);
DEF_PROMISE_TYPE(Fn::Odd,  collaz);

/*
// Declare a Harmonize program type capable of executing all promises in the set {Fn::Even,Fn::Odd}
// and which use the Global+Group+Thread states declared for 'ProgState'. For convenience,
// we name this type 'ProgType'.
*/
typedef  HarmonizeProgram < PromiseUnion<Fn::Even,Fn::Odd>, ProgState > ProgType;


/*
// Defines the async function to handle 'Even' promises for programs with type 'ProgType'. We
// furthermore specify that the name of the argument accepted by the function is 'arg'. Global,
// Group, and Thread states are respectively accessed thro
*/
DEF_ASYNC_FN(ProgType, Fn::Even, arg) {

	if( arg.val <= 1 ){
		global.output[arg.original] = arg.step;
		return;
	}

	arg.step += 1;
	arg.val  /= 2;

	if( (arg.val%2) == 0 ){
		ASYNC_CALL(Fn::Even,arg);
	} else {
		ASYNC_CALL(Fn::Odd, arg);
	}

}
/*
// Same as the definition above, but handling 'Odd' promises and performing the operations
// corresponding to odd values in the collaz iteration.
*/
DEF_ASYNC_FN(ProgType, Fn::Odd, arg) {

	if( arg.val <= 1 ){
		global.output[arg.original] = arg.step;
		return;
	}

	arg.step += 1;
	arg.val  *= 3;
	arg.val  += 1;

	if( (arg.val%2) == 0 ){
		ASYNC_CALL(Fn::Even,arg);
	} else {
		ASYNC_CALL(Fn::Odd, arg);
	}

}

/*
// Defines the initialization function for programs of type 'ProgType'. This function is called by
// all threads in all work groups just prior to beginning normal execution. States are accessible
// through the 'global', 'group', and 'thread' variables, just like when defining async functions.
// 
// Here, we initialize the work iterator to iterate over a range of integers unique to the work
// group, distributing the set of integers to process more or less evenly across the set of
// work groups.
*/
DEF_INITIALIZE(ProgType) {

	unsigned int group_data_size = (global.limit - global.start) / gridDim.x;
	unsigned int group_start = global.start + group_data_size * blockIdx.x;
	unsigned int group_end   = group_start + group_data_size;
	if( blockIdx.x == (gridDim.x-1) ){
		group_end = global.limit;
	}

	group.iterator.reset(group_start,group_end);

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
DEF_FINALIZE(ProgType) {


}



/*
// Defines the work making function for programs of type 'ProgType'. This function is called by
// work groups whenever they notice that they are running out of work to perform. To indicate
// that there is still more work to perform, return 'true'. To indicate that there is no more
// work left for the work group to make, return 'false', at which point, the work group will no
// longer call this function for the remainder of the execution run.
*/
DEF_MAKE_WORK(ProgType) {


	unsigned int index;
        if(!group.iterator.step(index)){
		return false;
	}

	collaz prom = { 0, index, index};
	if( (index % 2) == 0 ){
		ASYNC_CALL(Fn::Even,prom);
	} else {
		ASYNC_CALL(Fn::Odd,prom);
	}

	return true;

}


// A function to double-check our work
unsigned int checker(unsigned int val){
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

	util::ArgSet args(argc,argv);


	unsigned int wg_count    = args["wg_count"];
	unsigned int cycle_count = args["cycle_count"] | 65536u;

	util::Stopwatch watch;

	watch.start();

	// Declare a global state and initialize it with the information  it requires:
	GlobalState gs;

	// Define the range of values to process through the collaz iteration
	gs.start  = 0;
	gs.limit  = args["limit"];
	
	// Define a device-side buffer of type 'unsigned int' with size 'gs.limit'
	util::DevVec<unsigned int> dev_out = util::DevVec<unsigned int>(gs.limit);
	// Assign the address of the device-side buffer to the global state so that the program
	// can know where to put its output.
	gs.output = dev_out;

	// Declare and instance of type 'ProgType' with an arena size of 2^(20) with a global state
	// initialized to the value of our declared global state struct. The arena size of a
	// program determines how much extra space it has to store work if it cannot store
	// everything inside shared memory. If you are *certain* that no work will spill into
	// global memory, you may get some performance benefits by seting the arena size to zero.
	ProgType::Instance instance = ProgType::Instance(0x0,gs);
	cudaDeviceSynchronize();
	util::check_error();
	
	// Initialize the instance using 32 work groups
	init<ProgType>(instance,32);
	cudaDeviceSynchronize();
	util::check_error();

	// Execute the instance using 240 work groups, with each work group performing up to
	// 65536 promise executions per thread before halting. If all promises are exhausted
	// before this, the program exits early.
	exec<ProgType>(instance,wg_count,cycle_count);
	cudaDeviceSynchronize();
	util::check_error();	

	watch.stop();

	float msec = watch.ms_duration();

	// Retrieve the data from the device-side buffer into a host-side vector.
	std::vector<unsigned int> host_out(gs.limit);
	dev_out >> host_out;

	// Double-check that the program has accurately evaluated the iterations assigned to it.
	for(unsigned int i=0; i<gs.limit; i++){
		if( host_out[i] != checker(i) ){
			printf("Bad output for input %d. Output is %d when it should be %d\n",i,host_out[i],checker(i));
		}
	}

	printf("Runtime: %f\n",msec);

	return 0;

}

