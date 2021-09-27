#include <stdio.h>
#include <cstdint>
#include <vector>





struct neutron {

	float p_x;
	float p_y;
	float p_z;
	float m_x;
	float m_y;
	float m_z;
	float time;
	unsigned int seed;

};

struct sim_params {

	int	span;
	int	horizon;

	int	count_lim;
	
	neutron* old_data;

	unsigned long long int* halted;

	float	div_width;
	float	pos_limit;
	int	div_count;

	float	fission_x;
	float	capture_x;
	float	scatter_x;
	float	combine_x;

	float   time_limit;

};




__device__ unsigned int random_uint(unsigned int& rand_state){

	rand_state = (1103515245u * rand_state + 12345u) % 0x80000000;
	return rand_state;

}



__device__ float random_norm(unsigned int& rand_state){

	unsigned int val = random_uint(rand_state);

	return ( ( (int) (val%65537) ) - 32768 ) / 32768.0;

}


__device__ float random_unorm(unsigned int& rand_state){

	unsigned int val = random_uint(rand_state);

	return ( (int) (val%65537) ) / 65537.0;

}




__device__ float random_2D_iso(unsigned int& rand_state){

	float m,x,y;
	m = 2;
	while ( m > 1 ){
		x = random_norm(rand_state);
		y = random_norm(rand_state);
		m = sqrt(x*x+y*y);
	}
	return x / m;

}


__device__ void random_3D_iso_mom(neutron& n){


	float mu = random_norm(n.seed);
	float az = 2.0 * 3.14159 * random_unorm(n.seed);

	float c = sqrt(1.0 - mu*mu);
	n.m_y = cos(az) * c;
	n.m_z = sin(az) * c;
	n.m_x = mu;


}

__device__ int pos_to_idx(sim_params& params, float pos){

	return params.div_count + (int) floor(pos / params.div_width);

}


__device__ float euc_mod(float num, float den) {

	return num - abs(den)*floor(num/abs(den));

}




__device__ float clamp(float val, float low, float high){

	if( val < low ){
		return low;
	} else if ( val > high ){
		return high;
	}
	return val;

}




__device__ bool step_neutron(sim_params params, neutron& n){

	// Advance particle position
	float step = - logf( 1 - random_unorm(n.seed) ) / params.combine_x;
	
	bool halt = false;
	if( n.time + step > params.time_limit){
		step = params.time_limit - n.time;
		halt = true;
	}
	n.time += step;

	n.p_x += n.m_x * step;
	n.p_y += n.m_y * step;
	n.p_z += n.m_z * step;

	float dist = n.p_x;

	int index = pos_to_idx(params,dist);

	// Break upon exiting medium
	if( (index < 0) || (index >= params.div_count*2) ){
		return false;
	}

	// Sample the event that occurs
	float samp = random_unorm(n.seed);
	
	if ( halt ){
		atomicAdd(&params.halted[index],1);
		return false;
	}else if( samp <= (params.scatter_x/params.combine_x) ){
		random_3D_iso_mom(n);
		return true;
	} 
	else {
		return false;
	}

}


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



struct common_context{

	sim_params  params;

	bool	    show;
	bool	    csv;

	util::Stopwatch watch;

};



common_context common_initialize(util::ArgSet& args){


	common_context result;

	sim_params& params = result.params;

	params.span       = args["span"] | 32u;
	params.horizon    = args["hrzn"] | 32u;
	params.count_lim  = args["num"]  | 1000u;;
	
	params.time_limit = args["time"] | 1.0f;

	params.div_width  = args["res"]  | 1.0f;
	params.pos_limit  = args["size"] | 1.0f;
	params.div_count  = params.pos_limit/params.div_width;

	params.fission_x  = args["fx"]   | 0.0f;
	params.capture_x  = args["cx"]   | 0.0f;
	params.scatter_x  = args["sx"]   | 0.0f;
	params.combine_x  = params.fission_x + params.capture_x + params.scatter_x;

	result.show       = args["show"];
	result.csv        = args["csv"];


	result.watch.start();

	cudaMalloc( (void**) &params.old_data,  sizeof(neutron) * params.count_lim );

	int elem_count = params.div_count*2;


	cudaMalloc( (void**) &params.halted,   sizeof(unsigned long long int)   * elem_count );
	cudaMemset( params.halted, 0, sizeof(unsigned long long int) * elem_count );

	return result;

}



void common_finalize(common_context& context){


	sim_params& params = context.params;

	context.watch.stop();

        float msecTotal = context.watch.ms_duration();
	
	int elem_count = params.div_count*2;


	unsigned long long int* result_raw;
	float* result;

	if( context.show || context.csv ){

		result_raw = (unsigned long long int*) malloc(sizeof(unsigned long long int)*elem_count);
		result = (float*) malloc(sizeof(float)*elem_count);

		cudaMemcpy(
			(void*)result_raw,
			params.halted,
			sizeof(unsigned long long int)*elem_count,
			cudaMemcpyDeviceToHost
		);

		float sum = 0;
		for(unsigned int i=0; i<elem_count; i++){
			float value = ((float)result_raw[i])/ (((float)params.div_width)*((float)params.count_lim));
			sum += value * params.div_width;
			result[i] = value;
		}

		printf("\n\nSUM IS: %f\n\n",sum);


	}


	if( context.csv ){

		for(unsigned int i=0; i<elem_count; i++){
			if( i == 0 ){
				printf("%f",result[i]);
			} else {
				printf(",%f",result[i]);
			}
		}
		printf("\n");
		
		return;

	}


	if( context.show ){

		util::cli_graph(result,elem_count,100,20,-params.pos_limit,params.pos_limit);

	}

	
	if( context.show ){
		printf("%f\n",msecTotal);
	} else {
		printf("%f",msecTotal);
	}



}


