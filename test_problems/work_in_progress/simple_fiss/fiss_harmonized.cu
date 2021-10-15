

#define checkError  check_error


#include "fiss_common.cu"

using namespace util;




typedef ProgramStateDef<SimParams,VoidState,VoidState> ProgState;

enum class Fn { Neutron };

DEF_PROMISE_TYPE(Fn::Neutron, unsigned int);

typedef  HarmonizeProgram < PromiseUnion<Fn::Neutron>, ProgState > ProgType;


DEF_ASYNC_FN(ProgType, Fn::Neutron, arg) {


	Neutron n;
	n = global.neutron_buffer[arg];

	#ifndef PRE_INIT
	if ( n.time <= 0 ){
		n.p_x = 0;
		n.p_y = 0;
		n.p_z = 0;
		n.time = 0.0;
		random_3D_iso_mom(n);
	}
	#endif


	int result;
	for(int i=0; i < global.horizon; i++){
		result = step_neutron(global,n);
		if( result != 0 ){
			return;
		}
	}

	#ifdef IOBUFF
	global.neutron_io.input_ptr()[arg] = n;
	ASYNC_CALL(Fn::Neutron,arg);		
	#else
	global.neutron_buffer[arg] = n;
	ASYNC_CALL(Fn::Neutron,arg);		
	#endif

}


DEF_INITIALIZE(ProgType) {


}


DEF_FINALIZE(ProgType) {


}


DEF_MAKE_WORK(ProgType) {


	unsigned int index;

	Iter<unsigned int> iter = global.source_id_iter->leap(14u);

	while(iter.step(index)){
		Neutron n;
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

		global.neutron_buffer[index] = n;
		
		ASYNC_CALL(Fn::Neutron,index);
	}

	return !global.source_id_iter->done();

}



int main(int argc, char *argv[]){


	ArgSet args(argc,argv);

	unsigned int wg_count = args["wg_count"];

	CommonContext com(args);

	cudaDeviceSynchronize();
		
	checkError();
	
	ProgType::Instance instance = ProgType::Instance(0xFFFFF,com.params);
	cudaDeviceSynchronize();
	check_error();
	
	init<ProgType>(instance,wg_count);
	cudaDeviceSynchronize();
	check_error();

	exec<ProgType>(instance,wg_count,0xFFFFF);
	cudaDeviceSynchronize();
	check_error();
	
	return 0;

}

