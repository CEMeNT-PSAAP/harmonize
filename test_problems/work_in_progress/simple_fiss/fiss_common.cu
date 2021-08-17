#include <stdio.h>
#include <cstdint>
#include <vector>


#ifndef DEF_THUNK_SIZE

#ifndef DEF_WG_COUNT
	#define DEF_WG_COUNT	1
#endif

#ifndef DEF_WG_SIZE
	#define DEF_WG_SIZE	32
#endif

const unsigned int WG_COUNT = DEF_WG_COUNT;
const unsigned int WG_SIZE  = DEF_WG_SIZE;


void checkError(){

	cudaError_t status = cudaGetLastError();

	if(status != cudaSuccess){
		const char* err_str = cudaGetErrorString(status);
		printf("ERROR: \"%s\"\n",err_str);
	}

}

#endif




#ifdef NEUTRON_3D
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
#else
struct neutron {

	float pos;
	float mom;
	float weight;
	unsigned int seed;

};
#endif

struct sim_params {

	int	span;
	int	horizon;

	int	count_lim;
	int*	new_count;
	int*	old_count;

	int*	data_iter;	
	int*	exit_iter;
	
	neutron* new_data;
	neutron* old_data;

	#ifdef NEUTRON_3D
	unsigned long long int* halted;
	#else
	float*	fission;
	float*	capture;
	float* 	flux;
	#endif

	float	div_width;
	float	pos_limit;
	int	div_count;

	float	fission_x;
	float	capture_x;
	float	scatter_x;
	float	combine_x;

	#ifdef NEUTRON_3D
	float   time_limit;
	#else
	float	fiss_coef;
	bool	flux_on;
	#endif

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


#ifdef NEUTRON_3D
__device__ void random_3D_iso_mom(neutron& n){

	#if 0
	float m = 2;
	while ( m > 1 ){
		n.m_x = random_norm(n.seed);
		n.m_y = random_norm(n.seed);
		n.m_z = random_norm(n.seed);
		m = sqrt( n.m_x*n.m_x + n.m_y*n.m_y + n.m_z*n.m_z );
		if (m <= 0.01){
			m = 2;
		}
	}

	n.m_x /= m;
	n.m_y /= m;
	n.m_z /= m;

	#else

	float mu = random_norm(n.seed);
	float az = 2.0 * 3.14159 * random_unorm(n.seed);

	float c = sqrt(1.0 - mu*mu);
	n.m_y = cos(az) * c;
	n.m_z = sin(az) * c;
	n.m_x = mu;

	#endif

}
#endif

__device__ int pos_to_idx(sim_params& params, float pos){

	return params.div_count + (int) floor(pos / params.div_width);

}


__device__ float euc_mod(float num, float den) {

	/*
	if ( den < 0 ) {
		return num-floor(num/den);
	} else {
		return num-ceil (num/den);
	}
	*/
	return num - abs(den)*floor(num/abs(den));

}


__device__ void atomic_iterate(int* iter, int limit, int step_size, int& offset, int& width){

	__syncthreads();
	if ( threadIdx.x == 0 ) {
		offset  = atomicAdd(iter, step_size);

		if( offset >= limit ) {
			atomicAdd(iter, -step_size);
			width = 0;
		} else if ( offset > (limit-step_size) ) {
			width = limit - offset;
			atomicAdd( iter, width - step_size );
		} else {
			width = step_size;
		}

	}
	__syncthreads();

}


__device__ float clamp(float val, float low, float high){

	if( val < low ){
		return low;
	} else if ( val > high ){
		return high;
	}
	return val;

}


#ifndef NEUTRON_3D
__device__ void flux_pass(sim_params& params, float start, float final ) {

	if ( start > final ) {
		float temp = final;
		final = start;
		start = temp;
	}

	int true_start_cell = pos_to_idx(params, start);
	int start_cell = clamp(true_start_cell, 0, params.div_count*2-1);
	if( true_start_cell != start_cell ){
		start = -params.div_width;
	}

	int true_final_cell = pos_to_idx(params, final);
	int final_cell = clamp(true_final_cell, 0, params.div_count*2-1);
	if( true_final_cell != final_cell ){
		final = params.div_width;
	}

	if(start_cell == final_cell){
		atomicAdd(&params.flux[start_cell],final-start);
		return;
	}

	float start_sub = euc_mod(start,params.div_width);
	float start_len = (params.div_width - start_sub) / params.div_width;
	atomicAdd(&params.flux[start_cell],start_len);

	for ( int idx = start_cell + 1; idx < final_cell; idx++ ) {
		atomicAdd(&params.flux[idx],1.0);
	}

	if ( start_cell != final_cell ) {
		float final_sub = euc_mod(final,params.div_width);
		float final_len = final_sub / params.div_width;
		atomicAdd(&params.flux[final_cell],final_len);
	}

}
#endif


__device__ bool step_neutron(sim_params params, neutron& n){

	// Advance particle position
	float step = - logf( 1 - random_unorm(n.seed) ) / params.combine_x;
	
	#ifdef NEUTRON_3D

	bool halt = false;
	if( n.time + step > params.time_limit){
		step = params.time_limit - n.time;
		halt = true;
	}
	n.time += step;

	n.p_x += n.m_x * step;
	n.p_y += n.m_y * step;
	n.p_z += n.m_z * step;

	//float dist = sqrt( n.p_x*n.p_x + n.p_y*n.p_y + n.p_z*n.p_z );
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

	#else
	float old_pos = n.pos;
	n.pos  += n.mom * step;
		
	int index = pos_to_idx(params,n.pos);

	// Tally flux, if that option is active
	if(params.flux_on){
		flux_pass(params,old_pos,n.pos);
	}

	// Break upon exiting medium
	if( (index < 0) || (index >= params.div_count*2) ){
		return false;
	}

	// Sample the event that occurs
	float samp = random_unorm(n.seed);
	if( samp <= (params.scatter_x/params.combine_x) ){
		n.mom = random_2D_iso(n.seed);
		return true;
	} 
	else if ( samp <= ((params.scatter_x + params.fission_x)/params.combine_x) ) {
		atomicAdd(&params.fission[index],params.fiss_coef*n.weight);
		return true;
	}
	else {
		atomicAdd(&params.capture[index],n.weight);
		return false;
	}

	#endif

}


__global__ void sim_pass(sim_params params){

	__shared__ int offset;
	__shared__ int width;

	atomic_iterate(params.data_iter,*params.old_count,params.span, offset, width);

	while ( offset < *params.old_count ) {


		for ( int idx = threadIdx.x; idx < width; idx += WG_SIZE ) {
	
			unsigned int id = offset+idx;
			neutron n = params.old_data[id];

			#ifdef NEUTRON_3D
			if( n.time <= 0 ){
				n.p_x = 0.0;
				n.p_y = 0.0;
				n.p_z = 0.0;
				random_3D_iso_mom(n);
				n.time = 0.0;
			}
			#else
			if ( n.weight <= 0 ){
				n.pos = 0.0;
				n.mom = random_2D_iso(n.seed);
				n.weight = 1.0;
			}
			#endif

			bool alive = true;		
			while ( alive ){
				alive = step_neutron(params,n);			
			}
		
		}
	
		atomic_iterate(params.data_iter,*params.old_count,params.span, offset, width);

	}

	
	__syncthreads();
	if( threadIdx.x == 0 ){
		int exit_idx = atomicAdd(params.exit_iter, 1);
		if ( exit_idx == WG_COUNT ) {
			atomicExch(params.exit_iter,0);
		}
	}


}



__global__ void sim_init(sim_params params){


	__shared__ int offset;
	__shared__ int width;	

	atomic_iterate(params.old_count,params.count_lim,params.span, offset, width);

	while ( offset < params.count_lim ) {

		for ( int idx = threadIdx.x; idx < width; idx += WG_SIZE ) {
		
			unsigned int id = offset+idx;
			#ifdef NEUTRON_3D
			params.old_data[id].time = -1.0;
			#else
			params.old_data[id].weight = -1.0;
			#endif
			params.old_data[id].seed   = id;
		
		}

		atomic_iterate(params.old_count,params.count_lim,params.span,offset,width);

	}

	__syncthreads();
	if ( threadIdx.x == 0 ) {
		int exit_idx = atomicAdd(params.exit_iter, 1);
		if ( exit_idx == (WG_COUNT-1) ) {
			atomicExch(params.exit_iter,0);
			atomicExch(params.data_iter,0);
			int old_count_temp = atomicAdd(params.old_count,0);
			//printf("Final thread. Old count is %d\n",old_count_temp);
		}
	}
}









enum ArgType {
	F32,
	U32,
	Opt,
};

union ArgData {
	uint32_t	u32;
	float		f32;
	bool		opt;

	ArgData(uint32_t val) { u32 = val; }
	ArgData(float val)    { f32 = val; }
	ArgData(bool  val)    { opt = val; }

	void parse(char* str,ArgType type){
		
		int result = 1;
		
		switch(type){
			case U32:
			result = sscanf(str,"=%u",&u32);
			break;
			case F32:
			result = sscanf(str,"=%f",&f32);
			break;
			case Opt:
			if( strcmp(str,"=false") == 0 ){
				opt = false;
			} else if ( strcmp(str,"=true" ) == 0 ){
				opt = true;
			} else {
				printf("BRUH: '%s'\n\n",str);
			}
			break;
		}

		
		if( result <= 0 ){
			printf("Error parsing value for argument '%s'\n",str);
			std::exit(1);
		}

	}

};

struct Arg {
	char*		name;
	ArgType		type;
	ArgData		data;
	bool		must_give;

	Arg(char* n, ArgType t, ArgData d) : name(n), type(t), data(d) {}
	
	Arg(const char* n, ArgType t, ArgData d) : Arg((char*)n,t,d) {}
	
	Arg(char* n, ArgType t, ArgData d, bool m) : name(n), type(t), data(d), must_give(m) {}
	
	Arg(const char* n, ArgType t, ArgData d, bool m) : Arg((char*)n,t,d,m) {}

	bool scan(char* str){
		char* name_iter = name;
		char* str_iter  = str;
		while(*name_iter){
			if( *name_iter != *str_iter ) {
				return false;
			}
			name_iter++;
			str_iter++;
		}
		data.parse(str_iter,type);
		must_give = false;
		return true;
	}

	const char* type_str(){
		
		switch(type){
			case U32: return "u32";
			case F32: return "f32";
			case Opt: return "bool";
		}

		return "???";

	}


};



struct Arguments {

	std::vector<Arg> args;

	void get_fail(char* name) {
		printf("Failed to get value for argument '%s'.\n",name);
		exit(1);
	}

	float get_f32 (char* name) {
		for( Arg& a : args ){
			if( strcmp(a.name,name) == 0 ){
				//printf("Value of %s is %f\n",a.name,a.data.f32);
				return a.data.f32;
			}
		}
		get_fail(name);
		return 0.0;
	}

	uint32_t get_u32 (char* name) {
		for( Arg& a : args ){
			if( strcmp(a.name,name) == 0 ){
				//printf("Value of %s is %d\n",a.name,a.data.u32);
				return a.data.u32;
			}
		}
		get_fail(name);
		return 0;
	}

	bool get_opt (char* name) {
		for( Arg& a : args ){
			if( strcmp(a.name,name) == 0 ){
				//printf("Value of %s is %d\n",a.name,a.data.opt);
				return a.data.opt;
			}
		}
		get_fail(name);
		return false;
	}

	
	float get_f32 (const char* name) { return get_f32( (char*) name); }

	uint32_t get_u32 (const char* name) { return get_u32( (char*) name); }

	bool get_opt (const char* name) { return get_opt( (char*) name); }

	void scan_fail(char* name) {
		printf("Unknown command line argument '%s'.\n",name);
		printf("Valid inputs include:\n\n");
		for ( Arg& a : args ) {
			printf("\t- %s=<%s>\n",a.name,a.type_str());
		}
		printf("\n");
		exit(1);
	}


	void scan (int argc, char *argv[]) {
		for ( int i = 1; i < argc; i++ ) {
			bool hit = false;
			for ( Arg& a : args ) {
				if ( a.scan( argv[i] ) ) {
					hit = true;
					break;
				}
			}
			if ( !hit ) {
				scan_fail( argv[i] );
			}
		}
	}

	Arguments (std::initializer_list<Arg> a): args(a) {}


};




void graph(float* data, int size, int width, int height, float low, float high){

	const char* lookup[25] = {
		"⠀","⡀","⡄","⡆","⡇",
		"⢀","⣀","⣄","⣆","⣇",
		"⢠","⣠","⣤","⣦","⣧",
		"⢰","⣰","⣴","⣶","⣷",
		"⢸","⣸","⣼","⣾","⣿"
	};

	
	float max = 0;
	for( int i=0; i<size; i++){
		if( data[i] > max ){
			max = data[i];
		}
	}

	printf("Max is %f\n",max);

	int x_iter;
	float l_val, r_val;
	float last=0;

	printf("%7.5f_\n",max);
	for(int i=0; i<height; i++){
		float base = (height-i-1)*max/height;
		printf("%7.5f_",base);
		x_iter = 0;
		for(int j=0; j<width; j++){
			l_val = 0;
			r_val = 0;
			int l_limit = (j*2*size)/(width*2);
			int r_limit = ((j*2+1)*size)/(width*2);
			float count = 0.0;
			for(; x_iter < l_limit; x_iter++){
				l_val += data[x_iter];
				//printf("%f,",data[x_iter]);
				count += 1.0;
			}
			l_val = ( count == 0.0 ) ? last : l_val / count;
			last = l_val;
			count = 0.0;
			for(; x_iter < r_limit; x_iter++){
				r_val += data[x_iter];
				count += 1.0;
			}
			r_val = ( count == 0.0 ) ? last : r_val / count;
			last = r_val;
			l_val = ( l_val - base )/max*height*4;
			r_val = ( r_val - base )/max*height*4;
			int l_idx = (l_val <= 0.0) ? 0 : ( (l_val >= 4.0) ? 4 : l_val );
			int r_idx = (r_val <= 0.0) ? 0 : ( (r_val >= 4.0) ? 4 : r_val );
			int str_idx = r_idx*5+l_idx;
			/*
			if( (str_idx < 0) || (str_idx >= 25) ){
				printf("BAD! [%d](%f:%d,%f:%d) -> (%d)",j,l_val,l_idx,r_val,r_idx,str_idx);
			}
			*/
			printf("%s",lookup[str_idx]);
		}
		printf("\n");
	}

	int   rule_size = 8*width/2;
	char* rule_vals = new char[rule_size]; 
	memset(rule_vals,'\0',rule_size);

	printf("        ");
	for(int j=0; j<width; j+=2){
		float l_limit = low + ((high-low)/width)*j;
		sprintf(&rule_vals[(8*j/2)],"%7.3f",l_limit);
		printf("\\ ");
	}
	printf("\n");
	for(int i=0; i<7; i++){
		printf("        ");
		for(int j=0; j<width; j+=2){
			printf(" %c",rule_vals[(8*j/2)+i]);
		}
		printf("\n");
	}

	free(rule_vals);

}







struct common_context{

	sim_params  params;

	bool	    show;
	bool	    csv;
	
	cudaEvent_t start;
	cudaEvent_t stop;

};



common_context common_initialize(int argc, char *argv[]){


	common_context result;

	Arguments args = Arguments({
		Arg( "sx"   , F32, ArgData(0.0f)  ),
		Arg( "cx"   , F32, ArgData(0.0f)  ),
		Arg( "fx"   , F32, ArgData(0.0f)  ),
		Arg( "flux" , Opt, ArgData(false) ),
		Arg( "show" , Opt, ArgData(true)  ),
		Arg( "res"  , F32, ArgData(1.0f)  ),
		Arg( "size" , F32, ArgData(1.0f)  ),
		Arg( "v"    , F32, ArgData(1.0f)  ),
		Arg( "time" , F32, ArgData(1.0f)  ),
		Arg( "span" , U32, ArgData(32u)   ),
		Arg( "hrzn" , U32, ArgData(32u)   ),
		Arg( "num"  , U32, ArgData(1000u) ),
		Arg( "csv"  , Opt, ArgData(false) )
	});

	args.scan(argc,argv);

	sim_params& params = result.params;

	params.span = args.get_u32("span");
	params.horizon = args.get_u32("hrzn");
	params.count_lim = args.get_u32("num");
	

	#ifdef NEUTRON_3D
	params.time_limit = args.get_f32("time");
	#else
	params.fiss_coef = args.get_f32("v");
	params.flux_on   = args.get_opt("flux");
	#endif

	params.div_width = args.get_f32("res");
	params.pos_limit = args.get_f32("size");
	params.div_count = params.pos_limit/params.div_width;

	params.fission_x = args.get_f32("fx");
	params.capture_x = args.get_f32("cx");
	params.scatter_x = args.get_f32("sx");
	params.combine_x = params.fission_x + params.capture_x + params.scatter_x;

	result.show     = args.get_opt("show");
	result.csv      = args.get_opt("csv");



        cudaEventCreate( &result.start );
        cudaEventCreate( &result.stop  );
	cudaDeviceSynchronize();
	cudaEventRecord( result.start, NULL );


	cudaMalloc( (void**) &params.new_count, sizeof(int) );
	cudaMalloc( (void**) &params.old_count, sizeof(int) );

	cudaMalloc( (void**) &params.data_iter,	sizeof(int) );	
	cudaMalloc( (void**) &params.exit_iter, sizeof(int) );

	cudaMalloc( (void**) &params.new_data,  sizeof(neutron) * params.count_lim );
	cudaMalloc( (void**) &params.old_data,  sizeof(neutron) * params.count_lim );

	int elem_count = params.div_count*2;


	#ifdef NEUTRON_3D	
	cudaMalloc( (void**) &params.halted,   sizeof(unsigned long long int)   * elem_count );
	cudaMemset( params.halted, 0, sizeof(unsigned long long int) * elem_count );
	#else
	cudaMalloc( (void**) &params.fission,   sizeof(float)   * elem_count );
	cudaMalloc( (void**) &params.capture,   sizeof(float)   * elem_count );
	cudaMalloc( (void**) &params.flux,      sizeof(float)   * elem_count );
	float* zeros = new float[elem_count];
	for(int i=0; i<elem_count; i++){
		zeros[i] = 0.0;
	}
	cudaMemcpy( (void*)params.capture, (void*)zeros, sizeof(float)*elem_count, cudaMemcpyHostToDevice );
	cudaMemcpy( (void*)params.flux, (void*)zeros, sizeof(float)*elem_count, cudaMemcpyHostToDevice );
	free(zeros);
	#endif	


	int temp_exit_iter = 0;
	cudaMemcpy(
		(void*)params.exit_iter,
		(void*)&temp_exit_iter,
		sizeof(int),
		cudaMemcpyHostToDevice
	);

	return result;

}



void common_finalize(common_context& context){


	checkError();
	cudaEventRecord( context.stop, NULL );

	sim_params& params = context.params;

        // wait for the stop event to complete:
        cudaDeviceSynchronize( );
	
	checkError();
	
	cudaEventSynchronize( context.stop );

        float msecTotal = 0.0f;
        cudaEventElapsedTime( &msecTotal, context.start, context.stop );	
	
	int elem_count = params.div_count*2;


	#ifdef NEUTRON_3D


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

		/*
		for(unsigned int i=0; i<elem_count; i++){
			result[i] /= sum;
		}
		*/	

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

		graph(result,elem_count,100,20,-params.pos_limit,params.pos_limit);

	}


	#else	

	float* result;

	if( context.show || context.csv ){

		result = (float*) malloc(sizeof(float)*elem_count);

		cudaMemcpy(
			(void*)result,
			params.capture,
			sizeof(float)*elem_count,
			cudaMemcpyDeviceToHost
		);

		for(unsigned int i=0; i<elem_count; i++){
			result[i] /= params.div_width*params.count_lim;
		}

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

		graph(result,elem_count,100,20,-params.pos_limit,params.pos_limit);


		if(params.flux_on){
			cudaMemcpy(
				(void*)result,
				params.flux,
				sizeof(float)*elem_count,
				cudaMemcpyDeviceToHost
			);
			for(unsigned int i=0; i<elem_count; i++){
				result[i] /= params.div_width*params.count_lim*1000;
			}
			graph(result,elem_count,100,20,-params.pos_limit,params.pos_limit);
		}

	}

	#endif
	
	if( context.show ){
		printf("%f\n",msecTotal);
	} else {
		printf("%f",msecTotal);
	}



}


