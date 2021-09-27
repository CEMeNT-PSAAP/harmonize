#include "fiss_common.cu"





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



