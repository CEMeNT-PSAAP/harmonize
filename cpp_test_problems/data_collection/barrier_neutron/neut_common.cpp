#include <stdio.h>
#include <cstdint>
#include <vector>



//#define FILO

//#define ALLOC_CHECK

//#define LEVEL_CHECK

//#define TIMER 16

//#define BY_REF



//#define TTEVT

using namespace util;


struct Neutron;

__device__ void random_3D_iso_mom(Neutron& n);

struct Neutron {

	float p_x;
	float p_y;
	float p_z;
	//unsigned int padding[32];
	float m_x;
	float m_y;
	float m_z;
	#ifdef TTEVT
	float ttevt;
	#endif
	float time;
	unsigned int seed;
	unsigned int cohort;
	unsigned int step_count;

	float weight;


	#ifdef HARMONIZE
	#ifdef FILO
	unsigned int next;
	#endif
	#ifdef ALLOC_CHECK
	unsigned int checkout;
	#endif
	#endif


	Neutron() = default;
	
	__device__ Neutron(unsigned int s, float t, float x, float y, float z, float w)
		: seed(s)
		, time(t)
		, p_x(x)
		, p_y(y)
		, p_z(z)
		, weight(w)
		#ifdef TTEVT
		, ttevt(0)
		#endif

		#ifdef HARMONIZE
		#ifdef FILO
		, next(mem::Adr<unsigned int>::null)
		#endif
		#endif

	{
		random_3D_iso_mom(*this);
	}

	
	__device__ Neutron child()
	{
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
		#ifdef HARMONIZE
		#ifdef FILO
		result.next = mem::Adr<unsigned int>::null;
		#endif
		#endif
		#ifdef TTEVT
		result.ttevt = 0;
		#endif
		return result;
	}



};

#define PCB
#ifdef PCB

struct PopConBuffer {
	
	typedef util::mem::template Adr<unsigned int> AdrType;

	mem::MemPool<Neutron,unsigned int> neutron_pool;
	iter::AtomicIter<unsigned int>     carry_iter;
	float         prior_weight;
	unsigned int  prior_count;
	unsigned int  after_count;

	unsigned int* carry_fuse;
	unsigned int* carry_over;
	unsigned int* carry_mark;
	unsigned int  carry_size;

	PopConBuffer* next;


	#define breaker(x,y) if( x != y ) { return ( x > y ); }
	__device__ bool break_tie(Neutron& x, Neutron& y) {
		unsigned int self_seed  = x.seed;
		unsigned int other_seed = y.seed;
		unsigned int self_hash  = random_uint( self_seed);
		unsigned int other_hash = random_uint(other_seed);

		breaker(self_hash,other_hash);
		breaker(x.p_x,    y.p_x );
		breaker(x.p_y,    y.p_y );
		breaker(x.p_z,    y.p_z );
		breaker(x.m_x,    y.m_x );
		breaker(x.m_y,    y.m_y );
		breaker(x.m_z,    y.m_z );
		breaker(x.time,   y.time);
		breaker(x.weight, y.weight);
		#ifdef TTEVT
		breaker(x.ttevt,  y.ttevt);
		#endif
		return true;
	}
	

	__device__ unsigned int make_index(Neutron& neutron) {
		unsigned int self_seed  = neutron.seed;
		return random_uint( self_seed ) % carry_size;
	}

	__device__ void set_prior(unsigned int count) {
		unsigned int old = atomicExch(&prior_count,count);
		//printf("{OLD:%d}",old);
	}

	__device__ void add_prior() {
		int old = atomicAdd(&prior_count,1);
		//printf("[%p:%d+1]\n",this,old);
	}

	__device__ void resolve_prior() {
		int old = atomicSub(&prior_count,1);
		//printf("[%p:%d-1]\n",this,old);
		if ( (old-1) == 0 ) {
			resolve_phase();
		}
	}

	__device__ unsigned int get_after() {
		return atomicAdd(&after_count,0);
	}

	__device__ float take_prior_weight() {
		return atomicExch(&prior_weight,0);
	}

	__device__ void add_after() {
		unsigned int old = atomicAdd(&after_count,1);
		//printf("{%p:%d+1}\n",this,old);
	}

	__device__ bool resolve_after() {
		int old = atomicSub(&after_count,1);
		//printf("{%p:%d-1}\n",this,old);
		return ((old-1) == 0);
	}

	__device__ void resolve_phase(){
		__threadfence();
		unsigned int after_count = get_after();
		float        prior_mult  = take_prior_weight() / after_count;
		if( next != nullptr ){
			next->set_prior(after_count);
			printf("Set next (%p) of %p to %d\n",next,this,after_count);
		} else {
			//printf("Null next");
		}

		unsigned int hash = 0;
		for(int i=0; i<carry_size; i++){
			unsigned int index = carry_over[i];
			if( index != AdrType::null ){
				neutron_pool[index].weight *= prior_mult;
				hash ^= neutron_pool[index].seed;
			}	
		}
		__threadfence();
		printf("{%d}\n",hash);
		carry_iter.reset(0,carry_size);
		//printf("{reset}\n");
		__threadfence();
	}

	__device__ void set_next(PopConBuffer* next_pcb){
		next = next_pcb;
	}

	__device__ void merge(Neutron& a, Neutron& b) {
		if (!break_tie(a,b)) {
			a = b;
		}
	}

	__device__ bool blow_fuse(unsigned int index){
		unsigned int bit_index = index % 32;
		unsigned int val_index = index / 32;
		unsigned int mask = 1 << bit_index;
		unsigned int result = atomicOr( carry_fuse+val_index, mask );
		return ( ( result & mask ) == 0 );
	}

	__device__ bool is_blown(unsigned int index){
		unsigned int bit_index = index % 32;
		unsigned int val_index = index / 32;
		unsigned int mask = 1 << bit_index;
		unsigned int result = atomicOr( carry_fuse+val_index, 0 );
		return ( ( result & mask ) != 0 );
	}

	__device__ void give(Neutron neut) {
		unsigned int  seed     = neut.seed;
		unsigned int  index    = make_index(neut);
		unsigned int& target   = carry_over[index];

		unsigned int& target_mark = carry_mark[index];
		unsigned int self_hash  = random_uint(seed);
		unsigned int old = atomicMax(&target_mark,self_hash);
		if( old > self_hash ){
			__threadfence();
			resolve_prior();
		} else {
			bool verbose = false;
			int census = neut.time - 1;
			if( (census == 98) && (index == 3728) ) {
				//printf("($$$$$)");
				verbose = true;
			}

			unsigned int  neut_adr = AdrType::null;
			unsigned int  swap_adr = atomicExch(&target,AdrType::null);
			
			atomicAdd(&prior_weight,neut.weight);
			__threadfence();

			if( swap_adr != AdrType::null ){
				if (break_tie(neut,neutron_pool[swap_adr])) {
					neutron_pool[swap_adr] = neut;
				}
				neut_adr = swap_adr;
				swap_adr = AdrType::null;
			} else {
				neut_adr = neutron_pool.alloc_index(seed);
				if( neut_adr == AdrType::null ){
					printf("\n\nGOTCHA!\n\n");
				}
				neutron_pool[neut_adr] = neut;
			}
			__threadfence();

			//! Represents whether or not the slot being added to has previously
			//! been occupied.
			int try_count = 0;
			while( neut_adr != AdrType::null ){
				try_count ++;
				if( try_count > 10 ){
					printf("!!!OUCH!!!");
					break;
				}
				swap_adr = atomicExch(&target,AdrType::null);
				if( swap_adr == AdrType::null ){
					neut_adr = atomicExch(&target,neut_adr);
					continue;
				}
				__threadfence();
				Neutron neut  = neutron_pool[neut_adr];
				Neutron swap  = neutron_pool[swap_adr];
				bool neut_wins = break_tie(neut,swap);
				__threadfence();
				if (neut_wins) {
					neutron_pool.free(swap_adr,seed);
				} else {
					neutron_pool.free(neut_adr,seed);
					neut_adr = swap_adr;
				}
				__threadfence();
			}
			if( blow_fuse(index) ) {
				if( census >= 96 ){
					//printf("<%d:%d blown>\n",census,index);
				}
				add_after();
			}
			__threadfence();
			resolve_prior();
		}
	}


	//! Attempts to retrieve a neutron from the PCB, placing the retrieved value
	//! in the referenced destination value. Returns true if and only if a value
	//! was successfully retrieved.
	__device__ bool take(Neutron& dest) {
		unsigned int index;
		unsigned int try_count = 0;
		while( carry_iter.step(index) ){
			try_count ++;
			//printf("#%d#",index);
			int adr = atomicExch(&carry_over[index], AdrType::null);
			atomicExch(&carry_mark[index], 0);
			if( adr != AdrType::null ){
				if ( ! is_blown(index ) ){
					//printf("[?%d?]",index);
				}
				dest = neutron_pool[adr];
				int census = dest.time - 1;
				if( census >= 96 ){
					//printf("<%d:%d taken>\n",census,index);
				}
				if( resolve_after() ){
					//printf("Cleaning up pool %p\n",this);
					neutron_pool.serial_init();
					printf("Cleaned pool %p\n",this);
					prior_count = 0;
					for(int i=0; i<(carry_size/32+1);i++){
						carry_fuse[i] = 0;
					}

				}
				__threadfence();
				return true;
			} else if ( is_blown(index) ){
				//printf("[!%d!]",index);
			}
		}
		return false;
	}


	__host__ PopConBuffer(
		unsigned int size,
		unsigned int pool_size, 
		unsigned int arena_size,
		unsigned int prior
	)
		: carry_size(size)
		, neutron_pool(pool_size,arena_size)
		, prior_count(prior)
		, prior_weight(0)
	{}

	__host__ void host_init()
	{
		util::host::check_error();
		carry_iter   = iter::AtomicIter<unsigned int>(0,0);
		carry_fuse   = host::hardMalloc<unsigned int>(carry_size/32 + 1);
		carry_over   = host::hardMalloc<unsigned int>(carry_size);
		carry_mark   = host::hardMalloc<unsigned int>(carry_size);
		cudaMemset(carry_fuse,   0,(sizeof(unsigned int)*(carry_size/32+1)));
		cudaMemset(carry_mark,   0, sizeof(unsigned int)*carry_size);
		cudaMemset(carry_over,0xFF,(sizeof(unsigned int)*carry_size));
		after_count  = 0;
		next = nullptr;
		neutron_pool.host_init();
		cudaDeviceSynchronize();
		util::host::check_error();
	}

	__host__ void host_free()
	{
		host::auto_throw( cudaFree( carry_fuse ) );
		host::auto_throw( cudaFree( carry_over ) );
		host::auto_throw( cudaFree( carry_mark ) );
		neutron_pool.host_free();
	}


};
#endif



#ifdef PCB
const int PCB_COUNT = 3;
#endif


struct SimParams {

	int	span;
	int	horizon;

	int	source_count;
	iter::AtomicIter<unsigned int> *source_id_iter;

	int census_count;
	#ifdef PCB
	int cohort_count;
	int pcb_size;
	int pcb_count;
	int pcb_psize;
	int pcb_asize;
	PopConBuffer **pcb;
	#endif

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

	#ifdef LEVEL_CHECK
	int*    level_total;
	#endif


	#ifdef TIMER
	unsigned long long int*    timer;
	#endif

	bool is_async;	

};





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

enum StepResultType {
	CENSUS,
	SCATTER,
	CAPTURE,
	FISSION,
	LOSS
};

struct StepResult {
	StepResultType type;
	int  value;
	int  census;
};


__device__ StepResult step_neutron(SimParams params, Neutron& n){


	StepResult result;
	result.type = StepResultType::LOSS;

	#ifdef TTEVT
	unsigned int old_seed = n.seed;
	#endif

	// Advance particle position
	float step = - logf( 1 - random_unorm(n.seed) ) / params.combine_x;	

	#ifdef TTEVT
	if( n.ttevt > 0 ){
		step = n.ttevt;
		n.ttevt = 0;
	}
	#endif

	float time_mod  = euc_mod(n.time,params.time_limit);
	int next_census = (n.time) / params.time_limit;
	result.census = next_census;
	if( time_mod + step >= params.time_limit ){
		float cut_step = (params.time_limit - time_mod)+0.000001;
		#ifdef TTEVT
		n.ttevt = step - cut_step;
		#endif
		step = cut_step;
		result.type   = StepResultType::CENSUS;
		result.value  = next_census;
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

	// Break upon exiting medium
	if( (index < 0) || (index >= params.div_count*2) ){
		result.type   = StepResultType::LOSS;
		return result;
	}

	// Sample the event that occurs
	float samp = random_unorm(n.seed);

	
	if ( result.type == StepResultType::CENSUS ){
		#ifdef TTEVT
		n.seed = old_seed;
		#endif
		int offset = result.value * params.div_count * 2;
		atomicAdd(&params.halted[offset+index],n.weight);
		return result;
	} else if( samp < (params.scatter_x/params.combine_x) ){
		random_3D_iso_mom(n);
		result.type   = StepResultType::SCATTER;
		return result;
	} else if ( samp < ((params.scatter_x + params.fission_x) / params.combine_x) ) {
		result.type   = StepResultType::FISSION;
		result.value  = params.fiss_mult;
		return result;
	} else if ( samp < ((params.scatter_x + params.fission_x + params.capture_x) / params.combine_x) ) {
		if ( params.implicit_capture ) {
			if( original_weight < params.weight_limit ){
				result.type   = StepResultType::CAPTURE;
				return result;
			} else {
				result.type   = StepResultType::SCATTER;
				return result;
			}
		} else {
			result.type   = StepResultType::CAPTURE;
			return result;
		}
	} else {
		result.type   = StepResultType::CAPTURE;
		return result;
	}

}




struct CommonContext{

	SimParams  params;

	bool	    show;
	bool	    csv;
	bool	    value;

	host::DevBuf<iter::AtomicIter<unsigned int>> source_id_iter;
	host::DevBuf<float> halted;

	#ifdef PCB
	std::vector<host::DevObj<PopConBuffer>> pcb_list;
	host::DevBuf<PopConBuffer*> pcb_addresses;
	#endif

	
	host::DevObj<iter::IOBuffer<unsigned int>> neutron_io;

	#ifdef TIMER
	host::DevBuf<unsigned long long int> timer;
	#endif


	Stopwatch watch;

	CommonContext(cli::ArgSet& args)
		: neutron_io( args["io_cap"] | args["num"] | 1000u )
		#ifdef TIMER
		, timer((size_t) TIMER)
		#endif
	{
		params.span          = args["span"] | 32u;
		params.horizon       = args["hrzn"] | 32u;
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

		params.implicit_capture  = args["imp_cap"];
		params.weight_limit      = args["wlim"] | 0.0001f;
		
		params.is_async   = args["async"];

		value = args["value"];
		show  = args["show"];
		csv   = args["csv"];



		#ifdef PCB
		params.cohort_count = args["cohorts"]    | 1;
		params.census_count = args["census"]     | 1;
		params.pcb_count    = args["pcb_count"]  | 3;
		params.pcb_size     = args["pcb_size"]   | 128;
		params.pcb_psize    = args["pcb_psize"]  | 128;
		params.pcb_asize    = args["pcb_asize"]  | 4096;
		std::vector<PopConBuffer*> pointer_list;
		for(int i=0; i<params.pcb_count*params.cohort_count; i++){
			unsigned int prior = 0;
			if( i%3 == 0 ){
				prior = params.source_count;
			}
			pcb_list.push_back(host::DevObj<PopConBuffer>(
				params.pcb_size,
				params.pcb_psize,
				params.pcb_asize,
				prior
			));
		}
		for(int i=0; i<params.pcb_count * params.cohort_count; i++){
			int cycle_index = i/3;
			int phase_index = i%3;
			int phase_next  = cycle_index*3 + (phase_index+1)%3;
			pcb_list[i].host_copy().next = pcb_list[phase_next];
			pointer_list.push_back(pcb_list[i]);
			pcb_list[i].push_data();
		}
		pcb_addresses << pointer_list;
		params.pcb = pcb_addresses;
		#else
		params.census_count = 1;
		#endif

		watch.start();

		unsigned int pool_size = args["pool"] | 0x8000000u;
		

		int elem_count  = params.div_count*2;
		#ifdef PCB
		elem_count *= params.census_count;
		#endif
		halted.resize(elem_count);
		params.halted = halted;
		cudaMemset( halted, 0, sizeof(float) * elem_count );

		
		//cudaMemset( neutron_pool->arena, 0, sizeof(Neutron) * neutron_pool->arena_size.adr );


		source_id_iter<< iter::AtomicIter<unsigned int>(0,args["num"]);
		params.source_id_iter = source_id_iter;

		params.neutron_io = neutron_io;

		
		#ifdef TIMER
		params.timer = timer;
		cudaMemset( timer, 0, sizeof(unsigned long long int) * TIMER );
		#endif

	}


	~CommonContext(){

		watch.stop();

		float msecTotal = watch.ms_duration();
		
		int elem_count = params.div_count*2;


		std::vector<float> result;
		float y_min, y_max;
		float sums[params.census_count];
		for(int i=0; i<params.census_count; i++){
			sums[i] =0;
		}

		if( show || csv || value){

			halted >> result;
			
			for(unsigned int h=0; h<params.census_count; h++){
				for(unsigned int i=0; i<elem_count; i++){
					int index = h*elem_count+i;
					result[index] /= (float) params.source_count;
					sums[h] += result[index];
					result[index] /= (float) params.div_width;
					if( (i == 0) && (h==0) ){
						y_min = result[i];
						y_max = result[i];
						y_max = 0.2;
					}
					y_min = (result[index] < y_min) ? result[index] : y_min;
					y_max = (result[index] > y_max) ? result[index] : y_max;
				}
			}



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

			util::cli::GraphShape shape;
			shape.y_min  = y_min;
			shape.y_max  = y_max;
			shape.x_min  = -params.pos_limit;
			shape.x_max  =  params.pos_limit;
			shape.width  = 60;
			shape.height = 16;

			//util::cli_graph(result,elem_count,100,20,-params.pos_limit,params.pos_limit);
			for(int i=0; i<params.census_count; i++){
				util::cli::cli_graph(result.data()+elem_count*i,elem_count,shape,util::cli::Block2x2Fill);
			}
		}

		#ifdef TIMER
		printf("Program times:\n");
		std::vector<unsigned long long int> times;
		timer >> times;
		double total = times[0];
		for(unsigned int i=0; i<times.size(); i++){
			double the_time = times[i];
			double prop = 100.0 * (the_time / total);
			printf("T%d: %llu (~%f%)\n",i,times[i],prop );
		}
		#endif

		printf("\n%f\n",msecTotal);

	}


};


