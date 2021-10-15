#include <stdio.h>
#include <cstdint>
#include <vector>


//#define IOBUFF

struct Neutron;

__device__ void random_3D_iso_mom(Neutron& n);
__device__ unsigned int random_uint(unsigned int& rand_state);

struct Neutron {

	float p_x;
	float p_y;
	float p_z;
	float m_x;
	float m_y;
	float m_z;
	float time;
	unsigned int seed;

	Neutron() = default;
	
	__device__ Neutron(unsigned int s, float t, float x, float y, float z)
		: seed(s)
		, time(t)
		, p_x(x)
		, p_y(y)
		, p_z(z)
	{
		random_3D_iso_mom(*this);
	}

	
	__device__ Neutron(Neutron& parent)
		: seed(random_uint(parent.seed))
		, time(parent.time)
		, p_x(parent.p_x)
		, p_y(parent.p_y)
		, p_z(parent.p_z)
	{
		random_3D_iso_mom(*this);
	}

};

struct SimParams {

	int	span;
	int	horizon;

	int	source_count;
	util::AtomicIter<unsigned int>* source_id_iter;

	#ifdef IOBUFF
	util::IOBuffer<Neutron>* neutron_io;
	#else
	Neutron* neutron_buffer;
	#endif

	unsigned long long int* halted;

	float	div_width;
	float	pos_limit;
	int	div_count;

	float	fission_x;
	float	capture_x;
	float	scatter_x;
	float	combine_x;

	float   time_limit;

	int     fiss_mult;

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


__device__ void random_3D_iso_mom(Neutron& n){


	float mu = random_norm(n.seed);
	float az = 2.0 * 3.14159 * random_unorm(n.seed);

	float c = sqrt(1.0 - mu*mu);
	n.m_y = cos(az) * c;
	n.m_z = sin(az) * c;
	n.m_x = mu;


}

__device__ int pos_to_idx(SimParams& params, float pos){

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




__device__ int step_neutron(SimParams params, Neutron& n){

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
		return -1;
	}

	// Sample the event that occurs
	float samp = random_unorm(n.seed);
	
	if ( halt ){
		atomicAdd(&params.halted[index],1);
		return -1;
	}else if( samp <= (params.scatter_x/params.combine_x) ){
		random_3D_iso_mom(n);
		return 0;
	} 
	else {
		return params.fiss_mult;
	}

}




struct CommonContext{

	SimParams  params;

	bool	    show;
	bool	    csv;

	util::DevBuf<util::AtomicIter<unsigned int>> source_id_iter;
	util::DevBuf<unsigned long long int> halted;
	
	#ifdef IOBUFF
	util::DevObj<util::IOBuffer<Neutron>> neutron_io;
	#else
	util::DevBuf<Neutron> neutron_buffer;
	#endif

	util::Stopwatch watch;

	CommonContext(util::ArgSet& args)
		#ifdef IOBUFF
		: neutron_io( args["io_cap"] | args["num"] | 1000u )
		#else
		: neutron_buffer( args["num"] | 1000u )
		#endif
	{


		params.span       = args["span"] | 32u;
		params.horizon    = args["hrzn"] | 32u;
		params.source_count  = args["num"]  | 1000u;;

		params.time_limit = args["time"] | 1.0f;

		params.div_width  = args["res"]  | 1.0f;
		params.pos_limit  = args["size"] | 1.0f;
		params.div_count  = params.pos_limit/params.div_width;

		params.fission_x  = args["fx"]   | 0.0f;
		params.capture_x  = args["cx"]   | 0.0f;
		params.scatter_x  = args["sx"]   | 0.0f;
		params.combine_x  = params.fission_x + params.capture_x + params.scatter_x;

		params.fiss_mult  = args["mult"] | 2;

		show = args["show"];
		csv  = args["csv"];



		watch.start();


		int elem_count = params.div_count*2;
		halted.resize(elem_count);
		params.halted = halted;
		cudaMemset( halted, 0, sizeof(unsigned long long int) * elem_count );


		source_id_iter<< util::AtomicIter<unsigned int>(0,args["num"]);
		params.source_id_iter = source_id_iter;

		
		#ifdef IOBUFF
		params.neutron_io = neutron_io;
		#else
		params.neutron_buffer = neutron_buffer;
		#endif
	}


	~CommonContext(){

		watch.stop();

		float msecTotal = watch.ms_duration();
		
		int elem_count = params.div_count*2;


		unsigned long long int* result_raw;
		float* result;

		if( show || csv ){

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
				float value = ((float)result_raw[i])/ (((float)params.div_width)*((float)params.source_count));
				sum += value * params.div_width;
				result[i] = value;
			}

			printf("\n\nSUM IS: %f\n\n",sum);


		}


		if( csv ){

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


		if( show ){

			util::cli_graph(result,elem_count,100,20,-params.pos_limit,params.pos_limit);

		}

		
		if( show ){
			printf("%f\n",msecTotal);
		} else {
			printf("%f",msecTotal);
		}

	}


};


