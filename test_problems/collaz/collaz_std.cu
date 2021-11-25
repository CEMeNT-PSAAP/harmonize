
__global__ void collaz(unsigned int start, unsigned int end, unsigned int* output){

	for(unsigned int offset=blockIdx.x*blockDim.x+start; offset < end; offset+= gridDim.x*blockDim.x){

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
		output[original] = steps;
	}

}



// A function to double-check our work
unsigned int checker(unsigned long long int val){
	unsigned int res = 0;
	while(val > 1){
		res++;
		if( (val%2) == 0){
			val /=2;
		} else {
			val *=3;
			val +=1;
		}
	}
	return res;
}





int main(int argc, char* argv[]){

	util::ArgSet args(argc,argv);

	// Take command line arguments. Work group size defaults to 32.
	unsigned int wg_size  = args["wg_size"]  | 32u;
	unsigned int wg_count = args["wg_count"];
	unsigned int limit    = args["limit"]; 


	util::Stopwatch watch;

	if( ! watch.start() ){
		printf("A\n");
	}

	// Make device-side output buffer
	util::DevBuf<unsigned int> dev_out((size_t)limit);	
	cudaDeviceSynchronize();
	util::check_error();

	collaz<<<wg_count,wg_size>>>(0,limit,dev_out);
	cudaDeviceSynchronize();
	util::check_error();

	if( ! watch.stop() ){
		printf("B\n");
	}

	float msec = watch.ms_duration();
	
	// Get output from device
	std::vector<unsigned int> host_out;
	dev_out >> host_out;

	// Double-check that the program has accurately evaluated the iterations assigned to it.
	for(unsigned int i=0; i<limit; i++){
		if( host_out[i] != checker(i) ){
			printf("Bad output for input %d. Output is %d when it should be %d\n",i,host_out[i],checker(i));
		}
	}

	printf("Runtime: %f\n",msec);

	return 0;
}

