#include <stdio.h>


#ifndef DEF_SPAN
	#define DEF_SPAN 65536
#endif



#ifndef DEF_MULTIPROCESSOR_COUNT
	#define DEF_MULTIPROCESSOR_COUNT 	1
#endif

#ifndef DEF_WARP_SIZE
	#define DEF_WARP_SIZE 			32
#endif

#ifndef DEF_FUNCTION_ID_COUNT
	#define DEF_FUNCTION_ID_COUNT 		4
#endif

#ifndef DEF_THUNK_SIZE
	#define DEF_THUNK_SIZE 			4
#endif

#ifndef DEF_STACK_MODE
	#define DEF_STACK_MODE			0
#endif

#ifndef DEF_RETRY_LIMIT
	#define DEF_RETRY_LIMIT			8
#endif



/*
// This helps in determining how many warps and threads can reasonably be running simultaneously.
*/
const unsigned int MULTIPROCESSOR_COUNT	= DEF_MULTIPROCESSOR_COUNT;
const unsigned int WARP_SIZE		= DEF_WARP_SIZE;
const unsigned int TEAMS_PER_SM = 1u;
const unsigned int TEAM_COUNT = TEAMS_PER_SM * MULTIPROCESSOR_COUNT;


void checkError(){

	cudaError_t status = cudaGetLastError();

	if(status != cudaSuccess){
		const char* err_str = cudaGetErrorString(status);
		printf("ERROR: \"%s\"\n",err_str);
	}

}



__device__ unsigned int *step_counts;

__global__ void basic_collaz(unsigned int start, unsigned int end){

	for(unsigned int offset=blockIdx.x*WARP_SIZE+start; offset < end; offset+= TEAM_COUNT*WARP_SIZE){

		unsigned int original = offset+threadIdx.x;
		unsigned long long int val = original;
		unsigned int steps = 0;
		while(val > 1){
			if( (val % 2) == 0 ){
				val = val / 2;
			} else {
				val = val * 3 + 1;
			}
			steps++;
		}
		step_counts[original] = steps;
	}

}



unsigned int collaz(unsigned long long int val){

	unsigned int result = 0;
	while(val > 1){
		if( (val % 2) == 0 ){
			val = val / 2;
		} else {
			val = val * 3 + 1;
		}
		result++;
	}
	return result;

}



struct program_context{

	unsigned int* step_counts_ptr;
	cudaEvent_t start;
	cudaEvent_t stop;

};



const unsigned int span = DEF_SPAN;



program_context* initialize(){

	program_context* result = new program_context;

        cudaEventCreate( &result->start );
        cudaEventCreate( &result->stop  );
	cudaDeviceSynchronize();
	cudaEventRecord( result->start, NULL );


	cudaMalloc( (void**) &result->step_counts_ptr, sizeof(unsigned int)*span );	
	checkError();
	
	cudaError succ = cudaMemcpyToSymbol(step_counts,&result->step_counts_ptr,sizeof(unsigned int*));
	if(succ == cudaSuccess){
		//printf("\n\nInitialized step_counts\n\n\n");
	} else if(succ == cudaErrorInvalidSymbol){
		printf("\n\nInvalid symbol!\n\n\n");
	} else {
		printf("\n\nUh Oh\n\n\n");
	}

        cudaDeviceSynchronize( );

	checkError();
	//printf("About to do the thing\n");
	
	return result;

}



void finalize(program_context* program){

	checkError();
	cudaEventRecord( program->stop, NULL );

        // wait for the stop event to complete:
        cudaDeviceSynchronize( );
	
	//printf("Just did the thing.\n");
	checkError();
	
	cudaEventSynchronize( program->stop );

        float msecTotal = 0.0f;
        cudaEventElapsedTime( &msecTotal, program->start, program->stop );
	
	
	

	unsigned int* host_results = (unsigned int*) malloc(sizeof(unsigned int)*span);
	cudaMemcpy((void*)host_results,program->step_counts_ptr,sizeof(unsigned int)*span,cudaMemcpyDeviceToHost);

	bool any_failure = false;
	unsigned int avg = 0;
	for(unsigned int i=0; i<span; i++){
		//printf("%d\t:\t%d",i,host_results[i]);
		unsigned int val = collaz(i);
		avg += val;
		if( host_results[i] == val ){
			//printf("\tS\n");
			;
		} else {
			any_failure = true;
			//printf("\tF\n");
		}
	}
	if(any_failure){
		printf("Failure encountered\n");
	} else {
		//printf("No failure found\n");
	}
	avg /= span;
	//printf("Average iteration value is %d\n",avg);
	//printf("\nSpan: %d\tTEAM_COUNT: %d\tWARP_SIZE: %d\n\n",span,TEAM_COUNT,WARP_SIZE);
	printf("%f",msecTotal);

}


int main(){

	program_context* program = initialize();	
	basic_collaz<<<TEAM_COUNT,WARP_SIZE>>>(0,span);
	finalize(program);

	return 0;
}

