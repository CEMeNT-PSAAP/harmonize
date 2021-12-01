

#include "fiss_common.cu"

using namespace util;


#define DYN

typedef ProgramStateDef<SimParams,VoidState,VoidState> ProgState;

enum class Fn { Neutron };

DEF_PROMISE_TYPE(Fn::Neutron, unsigned int);

typedef  HarmonizeProgram < PromiseUnion<Fn::Neutron>, ProgState > ProgType;


DEF_ASYNC_FN(ProgType, Fn::Neutron, arg) {


	if( arg == Adr<unsigned int>::null ){
		printf("{   Bad argument!   }");
		return;
	}


	Neutron n;

	#ifdef DYN
	n = (*global.neutron_pool)[arg];
	#else
	n =  global.neutron_buffer[arg];
	#endif

	int result = 0;
	for(int i=0; i < global.horizon; i++){
		result = step_neutron(global,n);
		if( result != 0 ){
			break;
		}
	}

	#ifdef DYN

	#ifdef FILO
	unsigned int last = n.next;
	#endif

	for(int i=0; i<result; i++){
		Neutron new_neutron(n);

		#ifdef FILO
		new_neutron.next = last;
		#endif

		unsigned int index = global.neutron_pool->alloc(_thread_context.rand_state);
		if( index != Adr<unsigned int>::null ){
			unsigned int old = atomicCAS(&(*global.neutron_pool)[index].checkout,0u,1u);
			if( old != 0 ){
				printf("\n{Bad fiss alloc %d at %d}\n",old,index);
			}

			#ifdef FILO
			last = index;
			#endif

			(*global.neutron_pool)[index] = new_neutron;

			#ifdef FILO
			if( i == (result-1) ){
				ASYNC_CALL(Fn::Neutron,index);
			}
			#else
			ASYNC_CALL(Fn::Neutron,index);
			#endif
		} else {
			printf("{Fiss alloc fail}");
		}


	}



	if( result == 0 ) {
		(*global.neutron_pool)[arg] = n;
		ASYNC_CALL(Fn::Neutron,arg);		
	}
	else {

		#ifdef FILO
		if( (result < 0) && (n.next != Adr<unsigned int>::null) ){
			ASYNC_CALL(Fn::Neutron,n.next);
		}
		#endif
		unsigned int old = atomicCAS(&(*global.neutron_pool)[arg].checkout,1u,0u);
		if( old != 1 ){
			printf("{Bad dealloc %d at %d}",old,arg);
		}
		global.neutron_pool->free(arg,_thread_context.rand_state);
	}

	#else
	global.neutron_buffer[arg] = n;
	ASYNC_CALL(Fn::Neutron,arg);		
	#endif

}


DEF_INITIALIZE(ProgType) {


}


DEF_FINALIZE(ProgType) {

	

}


DEF_MAKE_WORK(ProgType) {


	unsigned int id;

	Iter<unsigned int> iter = global.source_id_iter->leap(8u);//(14u);

	while(iter.step(id)){
		Neutron n(id,0.0,0.0,0.0,0.0);

		#ifdef FILO
		n.next = Adr<unsigned int>::null;
		#endif

		#ifdef DYN
		unsigned int index = global.neutron_pool->alloc(_thread_context.rand_state);
		while(index == Adr<unsigned int>::null){
			printf("FAIL");
			index = global.neutron_pool->alloc(_thread_context.rand_state);
		}
		
		if( (index != Adr<unsigned int>::null) ) { //&& (index != 0) ){
			(*global.neutron_pool)[index] = n;
			unsigned int old = atomicCAS(&(*global.neutron_pool)[index].checkout,0u,1u);
			if( old != 0 ){
				printf("\n{Bad alloc %d at %d}\n",old,index);
			}
			ASYNC_CALL(Fn::Neutron,index);
		}
		#else
		global.neutron_buffer [id] = n;
		ASYNC_CALL(Fn::Neutron,id);
		#endif
		
	}

	return !global.source_id_iter->done();

}



int main(int argc, char *argv[]){


	ArgSet args(argc,argv);

	unsigned int wg_count = args["wg_count"];

	CommonContext com(args);

	cudaDeviceSynchronize();
		
	check_error();
	
	ProgType::Instance instance = ProgType::Instance(0xFFFFF,com.params);
	cudaDeviceSynchronize();
	check_error();

	init<ProgType>(instance,wg_count);
	cudaDeviceSynchronize();
	check_error();

	while(! instance.complete() ){
		exec<ProgType>(instance,wg_count,0xFFFFF);
		cudaDeviceSynchronize();
		check_error();
	}

	return 0;

}

