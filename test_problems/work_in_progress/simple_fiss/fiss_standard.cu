#include "fiss_common.cu"





typedef common_context program_context;


program_context* initialize(int argc, char *argv[]){


	program_context* result = new program_context;

	*result = common_initialize(argc,argv); 


        cudaDeviceSynchronize( );

	checkError();
	
	return result;

}



void finalize(program_context* program){

	common_finalize(*program);

}



int main(int argc, char *argv[]){

	

	program_context* program = initialize(argc,argv);	
	sim_init<<<WG_COUNT,WG_SIZE>>>(program->params);
        cudaDeviceSynchronize( );
	//printf("Initialized.\n");
	sim_pass<<<WG_COUNT,WG_SIZE>>>(program->params);
	finalize(program);

	return 0;
}

