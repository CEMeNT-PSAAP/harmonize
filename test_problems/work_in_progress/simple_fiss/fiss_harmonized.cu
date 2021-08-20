#include "fiss_common.cu"



//#define PRE_INIT



__device__ sim_params dev_params;



enum async_fn {
	NEUTRON=0,
};


struct union_thunk {

	union {

		ctx_thunk	thunk;

		#ifdef INDIRECT
		unsigned int	neu_index;
		#else
		neutron		neu_thunk;
		#endif

	};

};



__device__ void do_neutron(ctx_shared& shr, ctx_local& loc, union_thunk& thunk){

	neutron n;
	#ifdef INDIRECT
	n = dev_params.old_data[thunk.neu_index];
	#else
	n = thunk.neu_thunk;
	#endif


	#ifdef NEUTRON_3D

	#ifndef PRE_INIT
	if ( n.time <= 0 ){
		n.p_x = 0;
		n.p_y = 0;
		n.p_z = 0;
		n.time = 0.0;
		random_3D_iso_mom(n);
	}
	#endif

	#else
	if ( n.weight <= 0 ){
		n.pos = 0;
		n.weight = 1.0;
		n.mom = random_2D_iso(n.seed);
	}
	#endif

	for(int i=0; i < dev_params.horizon; i++){
		if( ! step_neutron(dev_params,n) ){
			return;
		}
	}

	#ifdef INDIRECT
	dev_params.old_data[thunk.neu_index] = n;
	#else
	thunk.neu_thunk = n;
	#endif	

	async_call(shr,loc, NEUTRON, 0, thunk.thunk);		

}








__device__ void make_work(ctx_shared& shr,ctx_local& loc){


	__shared__ unsigned long long int work_start;
	__shared__ unsigned long long int work_limit;
	__shared__ unsigned long long int end;

	if( current_leader() ){

		
		unsigned long long int work_space = WG_SIZE*(STASH_SIZE-2);
		const unsigned long long int work_width = dev_params.count_lim / WG_COUNT;
		unsigned long long int base = work_width * blockIdx.x;
		end  = base + work_width;

		if( blockIdx.x == (WG_COUNT-1)){
			end = dev_params.count_lim;
		}
		
		work_start = base + work_space * shr.work_iterator;

		if( work_start >= end ){
			work_start = end;
			work_limit = end;
		} else if ( (work_start + work_space) >= end ) {
			work_limit = end;
		} else {
			work_limit = work_start + work_space;
		}

		shr.work_iterator += 1;

	}

	__syncwarp();

	if( current_leader() && (work_limit >= end) ){
		shr.can_make_work = false;
	}

	ctx_thunk thunk;
	for(unsigned long long int offset = work_start; offset < work_limit; offset += WG_SIZE){

		unsigned long long int id = offset + threadIdx.x;

		if( id >= work_limit ) {
			break;
		}

		#ifdef NEUTRON_3D
		
		#ifdef INDIRECT
		thunk.data[0] = (unsigned int) id;
		neutron n;
		n.seed   = id;

		#ifdef PRE_INIT
		n.p_x = 0.0;
		n.p_y = 0.0;
		n.p_z = 0.0;
		random_3D_iso_mom(n);
		n.time = 0.0;
		#else
		n.time   = -1.0;
		#endif

		dev_params.old_data[id] = n;
		#else
		thunk.data[6] = (unsigned int) __float_as_uint(-1.0);
		thunk.data[7] = (unsigned int) id;	
		#endif	
		
		#else
		
		#ifdef INDIRECT
		thunk.data[0] = (unsigned int) id;
		neutron n;
		n.weight = -1.0;
		n.seed   = id;
		dev_params.old_data[id] = n;
		#else
		thunk.data[2] = (unsigned int) __float_as_uint(-1.0);
		thunk.data[3] = (unsigned int) id;	
		#endif	
		
		#endif
		
		async_call(shr,loc,NEUTRON,0,thunk);

	} 


	__syncwarp();


}




__device__ void do_async(ctx_shared& shr,ctx_local& loc, unsigned int func_id,ctx_thunk& thunk){

	/*
	// This switch is where one of the most abstract aspects of this system is handled,
	// namely diverting the flow of execution of work groups to the functions that are
	// required to process queued work. This is simply handled by switching to the function
	// by the function id that a respective link is labeled with.
	*/
	switch(func_id){
		case NEUTRON:
			do_neutron (shr,loc,*(union_thunk*)&thunk);
			break;
		
		default:
			/* 
			// REALLY BAD: We've been asked to execute a function
			// with an ID that we haven't heard of. In this case,
			// we set an error flag and halt.
			*/
			set_flags(shr,BAD_FUNC_ID_FLAG);
			shr.keep_running = false;
			break;
	}


}



struct program_context{

	sim_params* dev_params_ptr;

	common_context common;

};






program_context* initialize(runtime_context context, int argc, char *argv[]){

	program_context* result = new program_context;

	result->common = common_initialize(argc,argv);
	
	cudaError succ = cudaMemcpyToSymbol(dev_params, &result->common.params,sizeof(sim_params));
        
	cudaDeviceSynchronize( );

	checkError();

	return result;

}








void finalize(runtime_context runtime, program_context* program){

	common_finalize(program->common);

}






