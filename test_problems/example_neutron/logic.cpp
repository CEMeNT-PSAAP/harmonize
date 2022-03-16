#include <stdio.h>
#include <cstdint>
#include <vector>

using namespace util;


struct Neutron;

__device__ void random_3D_iso_mom(Neutron& n);

struct Neutron {

	float p_x;
	float p_y;
	float p_z;
	float m_x;
	float m_y;
	float m_z;
	float time;
	unsigned int seed;
	
	float weight;

	Neutron() = default;
	
	__device__ Neutron(unsigned int s, float t, float x, float y, float z, float w)
		: seed(s)
		, time(t)
		, p_x(x)
		, p_y(y)
		, p_z(z)
		, weight(w)

	{
		random_3D_iso_mom(*this);
	}

	// Used when fission occurs
	__device__ Neutron child()
	{
		// Make sure that the seed is thoroughly hashed
		Neutron result = *this;
		result.seed = random_uint(seed);
		result.seed ^= __float_as_uint(time);
		random_uint(result.seed);
		result.seed ^= __float_as_uint(p_x);
		random_uint(result.seed);
		result.seed ^= __float_as_uint(p_y);
		random_uint(result.seed);
		result.seed ^= __float_as_uint(p_z);
		random_uint(result.seed);
		random_3D_iso_mom(result);
		return result;
	}

};


struct DeviceState {

	int	horizon;

	int	source_count;
	iter::AtomicIter<unsigned int> *source_id_iter;

	iter::IOBuffer<unsigned int> *neutron_io;

	float* halted;

	float	div_width;
	float	pos_limit;
	int	div_count;

	float	fission_x;
	float	capture_x;
	float	scatter_x;
	float	combine_x;

	float   time_limit;

	bool    implicit_capture;
	float   weight_limit;

	int     fiss_mult;
	
};




// Used for generating a random float between -1 and 1, with uniform distribution
__device__ float random_norm(unsigned int& rand_state){

	unsigned int val = random_uint(rand_state);

	return ( ( (int) (val%65537) ) - 32768 ) / 32768.0;

}


// Used for generating a random float between 0 and 1, with uniform distribution
__device__ float random_unorm(unsigned int& rand_state){

	unsigned int val = random_uint(rand_state);

	return ( (int) (val%65537) ) / 65537.0;

}




// Used for generating a random 2D vector, with uniform distribution
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


// Sets neutron's momentum to random direction, with uniform distribution
__device__ void random_3D_iso_mom(Neutron& n){


	float mu = random_norm(n.seed);
	float az = 2.0 * 3.14159 * random_unorm(n.seed);

	float c = sqrt(1.0 - mu*mu);
	n.m_y = cos(az) * c;
	n.m_z = sin(az) * c;
	n.m_x = mu;


}

// Maps a neutron's x position to the correspoding tally index
__device__ int pos_to_idx(DeviceState& params, float pos){

	return params.div_count + (int) floor(pos / params.div_width);

}


// Euclidean modulus
__device__ float euc_mod(float num, float den) {

	return num - abs(den)*floor(num/abs(den));

}



// Clamps floating point values
__device__ float clamp(float val, float low, float high){

	if( val < low ){
		return low;
	} else if ( val > high ){
		return high;
	}
	return val;

}



// Iterates the neutron forward one step, returning -1 on capture, 0 on scatter, and a positive number
// on fission, with the positive number being the number of neutrons produced by the fission
__device__ int step_neutron(DeviceState params, Neutron& n){

	
	///////////////////////////////////////////////////////////////////////////////////////////
	// Advance particle position
	///////////////////////////////////////////////////////////////////////////////////////////
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

	float original_weight = n.weight;
	if ( params.implicit_capture && (n.weight >= params.weight_limit) ) {
		n.weight *= expf(-step*params.capture_x);
	}

	float dist = n.p_x;

	int index = pos_to_idx(params,dist);


	///////////////////////////////////////////////////////////////////////////////////////////
	// Break upon exiting medium
	///////////////////////////////////////////////////////////////////////////////////////////
	if( (index < 0) || (index >= params.div_count*2) ){
		return -1;
	}
	
	///////////////////////////////////////////////////////////////////////////////////////////
	// Sample the event that occurs, branching by event
	///////////////////////////////////////////////////////////////////////////////////////////
	float samp = random_unorm(n.seed);

	// If the time limit has been reached, tally the census and report capture
	if ( halt ){
		atomicAdd(&params.halted[index],n.weight);
		return -1;
	// On scatter, update the momentum
	} else if( samp < (params.scatter_x/params.combine_x) ){
		random_3D_iso_mom(n);
		return 0;
	// On fission, return the number of neutrons generated
	} else if ( samp < ((params.scatter_x + params.fission_x) / params.combine_x) ) {
		return params.fiss_mult;
	// On capture, check if implicit capture is active and weight is above weight limit.
	// If not, report capture, otherwise, update weight and report no change.
	} else if ( samp < ((params.scatter_x + params.fission_x + params.capture_x) / params.combine_x) ) {
		if ( params.implicit_capture ) {
			if( original_weight < params.weight_limit ){
				return -1;
			} else {
				return 0;
			}
		} else {
			return -1;
		}
	// If all else fails (it should not) return -1
	} else {
		return -1;
	}

}




