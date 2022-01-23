#include "neut_common.cpp"


using namespace util;





__global__ void sim_pass(SimParams params){


	while( ! params.source_id_iter->done() ) {

		iter::Iter<unsigned int> iter = params.source_id_iter->leap(1);
		
		unsigned int id;
		while ( iter.step(id) ){

			Neutron n(id,0.0,0.0,0.0,0.0,1.0);

			int result = 0;		
			while ( result == 0 ){
				result = step_neutron(params,n);			
			}

			if ( result == -1 ){
				break;
			}

		
		}

	}

}



int main(int argc, char *argv[]){

	cli::ArgSet args(argc,argv);

	unsigned int wg_count = args["wg_count"];
	unsigned int wg_size  = args["wg_size"] | 32u;

	CommonContext context(args); 
        cudaDeviceSynchronize( );
	host::check_error();

	#ifdef IOBUFF
	sim_pass<<<wg_count,wg_size>>>(context.params);
	#else
	sim_pass<<<wg_count,wg_size>>>(context.params);
	#endif

	return 0;

}



