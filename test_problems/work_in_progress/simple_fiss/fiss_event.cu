#include "fiss_common.cu"

using namespace util;


__global__ void sim_init(sim_params params){
	
	unsigned int group_data_size = params.count_lim / gridDim.x;
	unsigned int group_start = group_data_size * blockIdx.x;
	unsigned int group_end   = group_start + group_data_size;
	if( blockIdx.x == (gridDim.x-1) ){
		group_end = params.count_lim;
	}

	Iter<unsigned int> iter(group_start+threadIdx.x,group_end,blockDim.x);
	
	unsigned int id;
	while ( iter.step(id) ){

		params.neutron_buffer[id].time = -1.0;
	
	}

}

__global__ void sim_pass(sim_params params){
	
	unsigned int group_data_size = params.count_lim / gridDim.x;
	unsigned int group_start = group_data_size * blockIdx.x;
	unsigned int group_end   = group_start + group_data_size;
	if( blockIdx.x == (gridDim.x-1) ){
		group_end = params.count_lim;
	}

	Iter<unsigned int> iter(group_start+threadIdx.x,group_end,blockDim.x);
	
	unsigned int id;
	while ( iter.step(id) ){

		neutron n = params.neutron_buffer[id];
		if( n.time < 0 ){
			n.seed = id;
			n.p_x = 0.0;
			n.p_y = 0.0;
			n.p_z = 0.0;
			random_3D_iso_mom(n);
			n.time = 0.0;
		}

		if( step_neutron(params,n) ){
				
		}
	
	}

}


int main(int argc, char *argv[]){

	ArgSet args(argc,argv);

	unsigned int wg_count = args["wg_count"];
	unsigned int wg_size  = args["wg_size"] | 32u;

	CommonContext context(args);
	
	cudaMalloc( (void**) &context.params.new_data,  sizeof(neutron) * context.params.count_lim );
 
        cudaDeviceSynchronize( );
	check_error();
	
	sim_pass<<<wg_count,wg_size>>>(context.params,true);

	while(true){
		sim_pass<<<wg_count,wg_size>>>(context.params,false);
	}

	common_finalize(context);

	return 0;

}



