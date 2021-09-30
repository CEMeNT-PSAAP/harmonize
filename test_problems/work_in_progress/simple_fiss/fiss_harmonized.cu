

#define checkError  util::check_error


#include "fiss_common.cu"


struct GlobalState{
	unsigned int  start;
	unsigned int  limit;
	sim_params    params;
};

struct GroupState { util::GroupWorkIter<unsigned int> iterator; };

typedef ProgramStateDef<GlobalState,GroupState,VoidState> ProgState;

enum class Fn { Neutron };

DEF_PROMISE_TYPE(Fn::Neutron, unsigned int);

typedef  HarmonizeProgram < PromiseUnion<Fn::Neutron>, ProgState > ProgType;


DEF_ASYNC_FN(ProgType, Fn::Neutron, arg) {


	neutron n;
	n = global.params.old_data[arg];

	#ifndef PRE_INIT
	if ( n.time <= 0 ){
		n.p_x = 0;
		n.p_y = 0;
		n.p_z = 0;
		n.time = 0.0;
		random_3D_iso_mom(n);
	}
	#endif


	for(int i=0; i < global.params.horizon; i++){
		if( ! step_neutron(global.params,n) ){
			return;
		}
	}

	global.params.old_data[arg] = n;
	ASYNC_CALL(Fn::Neutron,arg);		



}


DEF_INITIALIZE(ProgType) {

	unsigned int group_data_size = (global.limit - global.start) / gridDim.x;
	unsigned int group_start = global.start + group_data_size * blockIdx.x;
	unsigned int group_end   = group_start + group_data_size;
	if( blockIdx.x == (gridDim.x-1) ){
		group_end = global.limit;
	}

	group.iterator.reset(group_start,group_end);


}


DEF_FINALIZE(ProgType) {


}


DEF_MAKE_WORK(ProgType) {


	unsigned int index;

	#if 1
	util::BasicIter<unsigned int> iter = group.iterator.multi_step<14>();

	while(iter.step(index)){
		neutron n;
		n.seed   = index;

		#ifdef PRE_INIT
		n.p_x = 0.0;
		n.p_y = 0.0;
		n.p_z = 0.0;
		random_3D_iso_mom(n);
		n.time = 0.0;
		#else
		n.time   = -1.0;
		#endif //PRE_INIT

		global.params.old_data[index] = n;
		
		ASYNC_CALL(Fn::Neutron,index);
	}
	#else

	group.iterator.step(index);

	#ifdef INDIRECT
	neutron n;
	n.seed   = index;

	#ifdef PRE_INIT
	n.p_x = 0.0;
	n.p_y = 0.0;
	n.p_z = 0.0;
	random_3D_iso_mom(n);
	n.time = 0.0;
	#else
	n.time   = -1.0;
	#endif //PRE_INIT

	global.params.old_data[index] = n;
	#else
	thunk.data[6] = (unsigned int) __float_as_uint(-1.0);
	thunk.data[7] = (unsigned int) id;	
	#endif //INDIRECT
	
	ASYNC_CALL(Fn::Neutron,index);


	#endif

	return !group.iterator.done();

}



int main(int argc, char *argv[]){


	util::ArgSet args(argc,argv);

	unsigned int wg_count = args["wg_count"];

	common_context com;

	com = common_initialize(args);
	cudaDeviceSynchronize();
		
	checkError();

	GlobalState gs;
	gs.start  = 0;
	gs.limit  = com.params.count_lim;
	gs.params = com.params;
	
	//printf("Making an instance...\n");
	ProgType::Instance instance = ProgType::Instance(0xFFFFF,gs);
	cudaDeviceSynchronize();
	util::check_error();
	
	//printf("Initing an instance...\n");
	init<ProgType>(instance,wg_count);
	cudaDeviceSynchronize();
	util::check_error();

	//printf("Execing an instance...\n");
	exec<ProgType>(instance,wg_count,0xFFFFF);
	cudaDeviceSynchronize();
	util::check_error();
	//printf("Finished exec.\n");
	

	common_finalize(com);

	return 0;

}

