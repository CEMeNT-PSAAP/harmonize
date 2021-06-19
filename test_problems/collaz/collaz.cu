#include "../../harmonize.cu"


#define MAKE_WORK


#ifndef DEF_SPAN
#define DEF_SPAN 65536
#endif


const unsigned int span = DEF_SPAN;


/*
// This will point to the buffer that output from threads will be written to.
*/
__device__ unsigned int*     step_counts;



//#define INNER_ITER_COUNT 112
#define INNER_ITER_COUNT 1024



#ifdef SPLIT_COLLAZ

enum async_fn {
	ROOT=0,
	COLLAZ_EVEN=1,
	COLLAZ_ODD =2,
};

#else 

enum async_fn {
	ROOT=0,
	COLLAZ=1,
};

#endif


struct root_thunk {
	unsigned int low;
	unsigned int limit;
};

struct collaz_thunk{
	unsigned long long int value;
	unsigned int original;
	unsigned int steps;
};

struct union_thunk {

	union {

		ctx_thunk	thunk;

		root_thunk	root;
		
		collaz_thunk	collaz;

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

		#ifdef SPLIT_COLLAZ
		
		thunk.collaz.value    = params.low;
		thunk.collaz.steps    = 0;
		thunk.collaz.original = params.low;
		
		if( (params.low % 2) == 0 ){
			async_call(shr,loc, COLLAZ_EVEN, 0, thunk.thunk);
		} else {
			async_call(shr,loc, COLLAZ_ODD , 0, thunk.thunk);
		}

		#else

		thunk.collaz.value    = params.low;
		thunk.collaz.steps    = 0;
		thunk.collaz.original = params.low;
		async_call(shr,loc, COLLAZ, 0, thunk.thunk);

		#endif

	}


}

__device__ void do_collaz(ctx_shared& shr, ctx_local& loc, union_thunk& thunk){

	collaz_thunk params = thunk.collaz;


	/*
	 else if ( params.value == params.original ){

		step_counts[params.original] = 0xFFFFFFFF;

	} else {
		step_counts[params.original] = params.steps;
	}
	*/

	for(int i=0; i<INNER_ITER_COUNT; i++){
		if(params.value <= 1){
			break;
		}
		if( (params.value%2) == 0){
			params.value = params.value / 2;
		} else {
			params.value = params.value * 3 + 1;
		}
		params.steps ++;
	}

	if(params.value <= 1){

		step_counts[params.original] = params.steps;
		db_printf("COMPLETED for %d\n",params.original);

		return;

	}

	thunk.collaz = params;

	#ifdef SPLIT_COLLAZ

	if( (params.value % 2) == 0 ) {
		async_call(shr,loc, COLLAZ_EVEN, 0, thunk.thunk);
	} else {
		async_call(shr,loc, COLLAZ_ODD , 0, thunk.thunk);
	}

	#else
	
	async_call(shr,loc, COLLAZ, 0, thunk.thunk);		

	#endif

}








__device__ void make_work(ctx_shared& shr,ctx_local& loc){

#ifdef MAKE_WORK	

	__shared__ unsigned long long int work_start;
	__shared__ unsigned long long int work_limit;
	__shared__ unsigned long long int end;

	if( current_leader() ){

		
		unsigned long long int work_space = WARP_SIZE*2;
		const unsigned long long int work_width = span / TEAM_COUNT;
		unsigned long long int base = work_width * blockIdx.x;
		end  = base + work_width;

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
	thunk.data[1] = 0;
	thunk.data[3] = 0;
	for(unsigned long long int offset = work_start; offset < work_limit; offset += WARP_SIZE){

		unsigned long long int start = offset + threadIdx.x;

		if( start >= work_limit ) {
			break;
		}

		thunk.data[0] = (unsigned int) start;
		thunk.data[2] = (unsigned int) start;	
	
		#ifdef SPLIT_COLLAZ
		if( (start%2) == 0 ){
			async_call(shr,loc,COLLAZ_EVEN,0,thunk);
		} else {
			async_call(shr,loc,COLLAZ_ODD ,0,thunk);
		}
		#else	
		async_call(shr,loc,COLLAZ,0,thunk);
		#endif			

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
		

#ifdef SPLIT_COLLAZ

		case COLLAZ_EVEN:
			do_collaz (shr,loc,*(union_thunk*)&thunk);
			break;

		case COLLAZ_ODD :
			do_collaz (shr,loc,*(union_thunk*)&thunk);
			break;

#else

		case COLLAZ:
			do_collaz (shr,loc,*(union_thunk*)&thunk);
			break;


#endif


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
	//printf("Queuing test call...\n");

	ctx_thunk thunk;
	unsigned int thunk_id;
	
	thunk_id = 0;
	thunk.data[0] = 0;
	thunk.data[1] = span;


	remote_call(runtime,thunk_id,thunk);

	cudaDeviceSynchronize();
	
	checkError();
	//runtime_overview(runtime);
	
	#endif

	return result;

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


unsigned int collaz_print(unsigned long long int val){

	unsigned int result = 0;
	while(val > 1){
		printf("%d->",val);
		if( (val % 2) == 0 ){
			val = val / 2;
		} else {
			val = val * 3 + 1;
		}
		result++;
	}
	printf("\n");
	return result;

}




void finalize(runtime_context runtime, program_context* program){

	cudaEventRecord( program->stop, NULL );

        // wait for the stop event to complete:
        cudaDeviceSynchronize( );
        cudaEventSynchronize( program->stop );

        float msecTotal = 0.0f;
        cudaEventElapsedTime( &msecTotal, program->start, program->stop );
	
	
	

	unsigned int* host_results = (unsigned int*) malloc(sizeof(unsigned int)*span);
	cudaMemcpy((void*)host_results,program->step_counts_ptr,sizeof(unsigned int)*span,cudaMemcpyDeviceToHost);

	bool any_failure = false;
	unsigned long int avg = 0;
	//printf("Span is: %d\n",span);
	unsigned int print_count = 0;
	for(unsigned int i=0; i<span; i++){
		//printf("%d\t:\t%d",i,host_results[i]);
		unsigned int val = collaz(i);
		avg += val;
		if( host_results[i] == val ){
			//printf("%d\t:\t%d",i,host_results[i]);
			//printf("\tS\n");
			;
		} else {
			any_failure = true;
			if(print_count < 1000){
				printf("%d\t:\t%d",i,host_results[i]);
				printf("\tF:%d\n",val);
				collaz_print(i);
				print_count++;
			}
			
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


	//printf("\n");
	//runtime_overview(runtime);

}






