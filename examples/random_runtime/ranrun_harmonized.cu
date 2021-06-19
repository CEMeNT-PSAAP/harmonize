#include "../../harmonize.cu"


#define MAKE_WORK



#ifndef DEF_THRESHOLD
#define DEF_THRESHOLD 8
#endif

const unsigned int THRESHOLD = DEF_THRESHOLD;

#ifndef DEF_SPAN
#define DEF_SPAN 65536
#endif


const unsigned int span = DEF_SPAN;


/*
// This will point to the buffer that output from threads will be written to.
*/
__device__ unsigned int*     step_counts;



//#define INNER_ITER_COUNT 12288
#ifndef INNER_ITER_COUNT
	#define INNER_ITER_COUNT 12288
#endif

enum async_fn {
	ROOT=0,
	RANRUN=1,
};


struct root_thunk {
	unsigned int low;
	unsigned int limit;
};

struct ranrun_thunk{
	unsigned int value;
	unsigned int original;
	unsigned int steps;
};

struct union_thunk {

	union {

		ctx_thunk	thunk;

		root_thunk	root;
		
		ranrun_thunk	ranrun;

	};

};



__device__ void root_init(ctx_shared& shr, ctx_local& loc, union_thunk& thunk){

	root_thunk params = thunk.root;

	if ( params.limit > (params.low+1) ){

		unsigned int mid = (params.low+params.limit)/2;

		thunk.root.low   = params.low;
		thunk.root.limit = mid;
		async_call(shr,loc, ROOT, 0, thunk.thunk);

		thunk.root.low   = mid;
		thunk.root.limit = params.limit;
		async_call(shr,loc, ROOT, 0, thunk.thunk);
		
	} else if ( params.limit == (params.low+1)) {

		thunk.ranrun.value    = params.low;
		thunk.ranrun.steps    = 0;
		thunk.ranrun.original = params.low;
		async_call(shr,loc, RANRUN, 0, thunk.thunk);

	}


}

__device__ void do_ranrun(ctx_shared& shr, ctx_local& loc, union_thunk& thunk){

	ranrun_thunk params = thunk.ranrun;

	//for(int k=0; k<10; k++){

		for(int i=0; i<INNER_ITER_COUNT; i++){
			if(params.value <= THRESHOLD){
				break;
			}
			params.value = (params.value * 999331 + 115249) % 0x80000000;
			params.steps ++;
		}
	//	if(active_count() < 32){
	//		break;
	//	}
	//}

	if(params.value <= THRESHOLD){

		step_counts[params.original] = params.steps;
		//printf("Finished for %d after %d steps\n",params.original,params.steps);
		return;

	}


	thunk.ranrun = params;

	async_call(shr,loc, RANRUN, 0, thunk.thunk);		

}








__device__ void make_work(ctx_shared& shr,ctx_local& loc){


#ifdef MAKE_WORK	

	__shared__ unsigned long long int work_start;
	__shared__ unsigned long long int work_limit;
	__shared__ unsigned long long int end;

	if( current_leader() ){

		
		unsigned long long int work_space = WARP_SIZE*(STASH_SIZE-1);
		const unsigned long long int work_width = span / TEAM_COUNT;
		unsigned long long int base = work_width * blockIdx.x;
		end  = base + work_width;

		if( blockIdx.x == (TEAM_COUNT-1)){
			end = span;
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
	thunk.data[2] = 0;
	for(unsigned long long int offset = work_start; offset < work_limit; offset += WARP_SIZE){

		unsigned long long int start = offset + threadIdx.x;

		if( start >= work_limit ) {
			break;
		}

		thunk.data[0] = (unsigned int) start;
		thunk.data[1] = (unsigned int) start;	
	
		async_call(shr,loc,RANRUN,0,thunk);

	} 


	__syncwarp();
#endif



}




__device__ void do_async(ctx_shared& shr,ctx_local& loc, unsigned int func_id,ctx_thunk& thunk){

	/*
	// This switch is where one of the most abstract aspects of this system is handled,
	// namely diverting the flow of execution of work groups to the functions that are
	// required to process queued work. This is simply handled by switching to the function
	// by the function id that a respective link is labeled with.
	*/
	switch(func_id){
		case ROOT:
			root_init (shr,loc,*(union_thunk*)&thunk);
			break;
		
		case RANRUN:
			do_ranrun (shr,loc,*(union_thunk*)&thunk);
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


/*
// END EXAMPLE CODE //////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
*/


struct program_context{

	unsigned int* step_counts_ptr;
	cudaEvent_t start;
	cudaEvent_t stop;

};






program_context* initialize(runtime_context runtime){

	program_context* result = new program_context;

        cudaEventCreate( &result->start );
        cudaEventCreate( &result->stop  );
	cudaDeviceSynchronize();
	cudaEventRecord( result->start, NULL );


	cudaMalloc( (void**) &result->step_counts_ptr, sizeof(unsigned int)*span );	
	
	cudaError succ = cudaMemcpyToSymbol(step_counts,&result->step_counts_ptr,sizeof(unsigned int*));
	checkError();


	#ifndef MAKE_WORK

	ctx_thunk thunk;
	unsigned int thunk_id;
	
	thunk_id = 0;
	thunk.data[0] = 0;
	thunk.data[1] = span;


	remote_call(runtime,thunk_id,thunk);

	cudaDeviceSynchronize();
	
	checkError();
	
	#endif

	return result;

}






unsigned int canon_ranrun(unsigned long long int val){

	unsigned int result = 0;
	while(val > THRESHOLD){
		val = (val * 999331 + 115249) % 0x80000000;
		result++;
	}
	return result;

}




void finalize(runtime_context runtime, program_context* program){

	cudaEventRecord( program->stop, NULL );

        // wait for the stop event to complete:
        cudaDeviceSynchronize( );
        cudaEventSynchronize( program->stop );

        float msecTotal = 0.0f;
        cudaEventElapsedTime( &msecTotal, program->start, program->stop );
	
	
	
	//*
	unsigned int* host_results = (unsigned int*) malloc(sizeof(unsigned int)*span);
	cudaMemcpy((void*)host_results,program->step_counts_ptr,sizeof(unsigned int)*span,cudaMemcpyDeviceToHost);

	bool any_failure = false;
	float avg = 0;
	float avg_dev = 0;

	unsigned int print_count = 0;
	for(unsigned int i=0; i<span; i++){
		unsigned int val = canon_ranrun(i);
		avg += val;
		if( host_results[i] == val ){
			;//printf("%d\n",val);
		} else {
			any_failure = true;
			if(print_count < 1000){
				printf("%d\t:\t%d",i,host_results[i]);
				printf("\tF:%d\n",val);
				print_count++;
			}
			
		}
	}

	avg /= span;

	for(unsigned int i=0; i<span; i++){
		unsigned int val = canon_ranrun(i);
		float dev = val - avg;
		dev = (dev < 0) ? -dev : dev;
		avg_dev += dev;
	}

	avg_dev /= span;

	printf("Average is: %f\tAverage deviation from mean is: %f\n",avg,avg_dev);

	if(any_failure){
		printf("Failure encountered\n");
	} else {
		printf("No failure found\n");
	}
	// */

	printf("%f\n",msecTotal);

}






