#include "fiss_common.cu"




__global__ void sim_pass(sim_params params){
	
	unsigned int group_data_size = params.count_lim / gridDim.x;
	unsigned int group_start = group_data_size * blockIdx.x;
	unsigned int group_end   = group_start + group_data_size;
	if( blockIdx.x == (gridDim.x-1) ){
		group_end = params.count_lim;
	}

	util::BasicIter<unsigned int> iter(group_start+threadIdx.x,group_end,blockDim.x);
	
	unsigned int id;
	while ( iter.step(id) ){

		neutron n;// = params.old_data[id];
		n.seed = id;
		n.p_x = 0.0;
		n.p_y = 0.0;
		n.p_z = 0.0;
		random_3D_iso_mom(n);
		n.time = 0.0;

		bool alive = true;		
		while ( alive ){
			alive = step_neutron(params,n);			
		}
	
	}

}


int main(int argc, char *argv[]){

	util::ArgSet args(argc,argv);

	unsigned int wg_count = args["wg_count"];
	unsigned int wg_size  = args["wg_size"] | 32u;

	common_context context = common_initialize(args); 
        cudaDeviceSynchronize( );
	util::check_error();
	
	sim_pass<<<wg_count,wg_size>>>(context.params);

	common_finalize(context);

	return 0;

}



